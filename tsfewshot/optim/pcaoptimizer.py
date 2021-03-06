import logging
import math
from typing import Dict, Union

import numpy as np
import torch
from torch import linalg, optim

from tsfewshot.config import Config

LOGGER = logging.getLogger(__name__)


class PCAOptimizer(optim.Optimizer):
    """Stochastic Gradient Descent optimizer with PCA transformation.

    Parameters
    ----------
    cfg : Config
        Run configuration.
    params : Dict[str, torch.Tensor]
        Dict of model names and parameters to optimize
    lr : float
        Learning rate
    pca : Union[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]
        PCA loaded from disk, as generated by ``pca.py``. Keys 'v' and 'explained_variance'.
        If `pca` is a nested dict, the outer keys must be parameter names and the inner keys per-parameter PCA dicts.
    """

    def __init__(self, cfg: Config,
                 optimizer_type: str,
                 params: Dict[str, torch.Tensor],
                 lr: float,
                 pca: Union[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]):
        super().__init__(params.values(), {'lr': lr})

        if len(self.param_groups) > 1:
            # Not sure if the order of parameters would remain consistent with the order in the PCA transformation
            # and if we'd always use the right learning rate.
            raise ValueError('PCAOptimizer supports only one param group')

        self._cfg = cfg
        self._params = params
        # '__'-keys in the layerwise PCA dict contain metadata
        first_pca_key = [k for k in pca.keys() if not k.startswith('__')][0]
        if isinstance(pca[first_pca_key], torch.Tensor):
            # PCA across all parameters
            self._pca: Dict[str, Dict[str, torch.Tensor]] = {'': pca}  # type: ignore
            self._layerwise_pca = False
        else:
            # per-parameter PCA
            self._pca: Dict[str, Dict[str, torch.Tensor]] = {k: v for k, v in pca.items()  # type: ignore
                                                             if not k.startswith('__')}
            self._layerwise_pca = True
        self._transformation: Dict[str, torch.Tensor] = {}
        self._explained_variance: Dict[str, torch.Tensor] = {}

        # we'll need a vector of gradients for the transformation, so we create it once to be faster during steps
        if self._layerwise_pca:
            LOGGER.info('Using layerwise PCA.')
            if any(k not in self._pca.keys() for k in self._params.keys()):
                raise ValueError('Must provide one PCA for each parameter.')
            if any(k not in self._params.keys() for k in self._pca.keys() if not k.startswith('__')):
                # ignore keys starting with __, which we use to store metadata of the PCA (see pca.py)
                LOGGER.warning('Some PCAs will remain unused. Are you sure this is the right PCA dictionary?')

        self._optimizer = optimizer_type.lower()
        self._interpolation_factor = cfg.pca_interpolation_factor
        if self._interpolation_factor is not None and not cfg.pca_normalize:
            LOGGER.warning('When interpolating between Adam/SGD and PCA, normalization should be active '
                           'to ensure similar learning rates are needed.')
        if self._optimizer in ['adam-pcaspace', 'adam-pcaspace-squared'] and self._interpolation_factor is not None:
            raise ValueError('Cannot interpolate between Adam and PCAspace-Adam.')

        if self._optimizer not in ['sgd', 'adam', 'adam-pcaspace', 'adam-pcaspace-squared',
                                   'sgd-squared', 'adam-squared']:
            raise ValueError(f'Unknown optimizer type {optimizer_type}.')

        self._initialize_pca()

        self._eps = 1e-8
        self._step = 0
        self._betas = (0.9, 0.999)
        self._mass = {}
        self._velocity = {}
        for param_name in self._pca.keys():
            n_params = None
            if self._optimizer in ['adam-pcaspace', 'adam-pcaspace-squared']:
                n_params = self._transformation[param_name].shape[1]
            elif self._optimizer in ['adam', 'adam-squared']:
                n_params = self._transformation[param_name].shape[0]
            if n_params is not None:
                self._mass[param_name] = torch.zeros(n_params).to(self.param_groups[0]['params'][0].device)
                self._velocity[param_name] = torch.zeros(n_params).to(self._mass[param_name].device)

    @torch.no_grad()
    def step(self):
        # note this will fail if a parameter doesn't get gradients. Since this shouldn't happen in our case,
        # we don't check if p.grad is None for performance reasons
        if self._layerwise_pca:
            gradient_vector = {param_name: param.grad.data.view(-1) for param_name, param in self._params.items()}
        else:
            gradient_vector = {'': torch.cat([p.grad.data.view(-1) for p in self._params.values()])}

        self._step += 1
        transformed_gradients = {}
        for param_name, grad in gradient_vector.items():
            param_transformation = self._transformation[param_name]
            param_explained_var = self._explained_variance[param_name]

            if self._optimizer in ['adam-pcaspace', 'adam-pcaspace-squared']:
                # Downproject
                transformed_gradient = param_transformation.T.mv(grad)
            elif self._interpolation_factor is not None:
                transformed_gradient = grad
            else:
                transformed_gradient = param_transformation.mv(param_explained_var * param_transformation.T.mv(grad))

            if self._optimizer in ['adam', 'adam-squared', 'adam-pcaspace', 'adam-pcaspace-squared']:
                # adapted from https://github.com/ganguli-lab/degrees-of-freedom/blob/main/lottery_subspace.py
                # Approximation of 1st and 2nd moment via exponential averaging
                self._mass[param_name] = self._betas[0] * self._mass[param_name] \
                    + (1.0 - self._betas[0]) * transformed_gradient
                self._velocity[param_name] = self._betas[1] * self._velocity[param_name] \
                    + (1.0 - self._betas[1]) * (transformed_gradient**2.0)

                # Bias correction
                hat_mass = self._mass[param_name] / (1.0 - (self._betas[0]**self._step))
                hat_velocity = self._velocity[param_name] / (1.0 - (self._betas[1]**self._step))

                # Update
                transformed_gradient = hat_mass / (torch.sqrt(hat_velocity) + self._eps)

                if self._optimizer in ['adam-pcaspace', 'adam-pcaspace-squared']:
                    # Undo projection
                    transformed_gradient = param_transformation.mv(param_explained_var * transformed_gradient)

            if self._interpolation_factor is not None:
                transformed_gradient = (1 - self._interpolation_factor) * transformed_gradient \
                    + self._interpolation_factor \
                    * param_transformation.mv(param_explained_var * param_transformation.T.mv(transformed_gradient))

            transformed_gradients[param_name] = transformed_gradient

        idx = 0
        learning_rate = self.param_groups[0]['lr']
        for param_name, param in self._params.items():
            param_size = param.numel()
            if self._layerwise_pca:
                grad = transformed_gradients[param_name]
            else:
                grad = transformed_gradients[''][idx:idx + param_size]
                idx += param_size
            param.data.sub_(grad.view(param.shape), alpha=learning_rate)

    def _initialize_pca(self):
        """Initialize the PCA values loaded from disk. """

        for param_name in self._pca.keys():
            device = self._params[list(self._params.keys())[0]].device
            transformation = self._pca[param_name]['v'].to(device)
            if self._optimizer not in ['sgd-squared', 'adam-pcaspace-squared', 'adam-squared']:
                explained_variance = self._pca[param_name]['explained_variance'].to(device)
            else:
                LOGGER.info('Calculating explained variance')
                explained_variance = (self._pca[param_name]['s']**2) / (self._pca[param_name]['u'].shape[0] - 1)
                explained_variance = explained_variance / explained_variance.sum()
                explained_variance = explained_variance.to(device)

            if not self._cfg.use_pca_weights:
                LOGGER.info('Ignoring PCA explained variance.')
                explained_variance = torch.ones_like(explained_variance)

            if self._cfg.pca_sparsity is not None:
                method, sparsity = self._cfg.pca_sparsity
                n_components = transformation.shape[1]
                if method == 'fraction-01':
                    n_weights = transformation.shape[0]
                    n_remaining = math.ceil((1 - sparsity) * n_weights)
                elif method == 'keep-n':
                    n_remaining = sparsity
                else:
                    raise ValueError(f'Sparsity method {method} not implemented for PCA-based finetuning.')
                if n_remaining > n_components:
                    LOGGER.warning('Fraction of remaining weights larger than number of PCA components.')
                    n_remaining = n_components

                LOGGER.info(f'Keeping top-{n_remaining} of {n_components} {param_name} components. '
                            f'These explain {explained_variance[:n_remaining].sum():.5f} of the variance.')

                transformation = transformation[:, :n_remaining]
                explained_variance = explained_variance[:n_remaining]

            if self._cfg.pca_normalize:
                # normalize such that the step size will be equal to plain gradient descent
                # (i.e., update step length of sqrt(number of model parameters))
                norm = linalg.norm(explained_variance, ord=2)
                factor = np.sqrt(transformation.shape[0]) / norm
                explained_variance = explained_variance * factor
                LOGGER.info(f'Normalizing {param_name} PCA with factor {factor:.5f} (original norm was {norm}).')

            self._transformation[param_name] = transformation
            self._explained_variance[param_name] = explained_variance
