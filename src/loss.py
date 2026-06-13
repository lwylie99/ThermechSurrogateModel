from dataclasses import dataclass, asdict

import torch
from torch import Tensor
from torch.autograd import grad

from components import PartSet, Component


def gradients(y, x, create_graph=True, retain_graph=True):
    return grad(y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=retain_graph
    )

def jacobian(u, coords):
    ''' Inputs: u = temp prediction (N, 1), coords[x, y] = spatial grid (N, 2)
        Outputs: [du/dx, du/dy] = temperature flux (N, 2)
    '''
    return gradients(u, coords)[0]

def laplacian_jacobian(u, coords, k=1):
    ''' k is constant conductivity '''
    jac = jacobian(u, coords)
    uxx = gradients(jac[..., 0] * k, coords)[0][..., 0]
    uyy = gradients(jac[..., 1] * k, coords)[0][..., 1]

    return jac, uxx + uyy

def residual_mse(residual) -> Tensor:
    return torch.nn.functional.mse_loss(residual, torch.zeros_like(residual))

def paired_loss(pred, act) -> Tensor:
    return torch.nn.functional.mse_loss(pred, act)