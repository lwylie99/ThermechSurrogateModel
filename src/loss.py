from typing import List

import numpy as np
import torch
from torch.autograd import grad

from components_thermal import BoundaryCondition, Robin
from src.mediums import Medium, Grid
from src.components_thermal import Gaussian, PdeCore


def gradients(y, x, create_graph=True, retain_graph=True):
    return grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=retain_graph,
    )


def jacobian(u, coords):
    return gradients(u, coords)[0]


def laplacian(u, coords, conductivity=1):
    jac = jacobian(u, coords)
    ux, uy = jac[..., 0, 0], jac[..., 0, 1]
    uxx = gradients(ux * conductivity, coords)[0][..., 0]
    uyy = gradients(uy * conductivity, coords)[0][..., 1]

    return ux, uy, uxx + uyy


def laplacian_jacobian(u, coords, conductivity=1):
    jac = jacobian(u, coords)
    uxx = gradients(jac[..., 0] * conductivity, coords)[0][..., 0]
    uyy = gradients(jac[..., 1] * conductivity, coords)[0][..., 1]

    return jac, uxx + uyy

def residual_mse(residual):
    torch.nn.functional.mse_loss(residual, torch.zeros_like(residual))

