from dataclasses import dataclass, asdict

import torch
from torch import Tensor
from torch.autograd import grad

from components import PartSet, Component


def gradients(y, x, create_graph=True, retain_graph=True):
    return grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=retain_graph
    )


def jacobian(u, coords):
    return gradients(u, coords)[0]

def laplacian_jacobian(u, coords, conductivity=1):
    jac = jacobian(u, coords)
    uxx = gradients(jac[..., 0] * conductivity, coords)[0][..., 0]
    uyy = gradients(jac[..., 1] * conductivity, coords)[0][..., 1]

    return jac, uxx + uyy


def residual_mse(residual) -> Tensor:
    return torch.nn.functional.mse_loss(residual, torch.zeros_like(residual))


@dataclass
class PartLoss(PartSet):
    top: Tensor = None
    bottom: Tensor = None
    left: Tensor = None
    right: Tensor = None
    core: Tensor = None
    paired: Tensor = None
    total: Tensor = None


@dataclass
class WeightedLoss(Component):
    # TODO: add weighting to loss
    initial: float = 0.0
    edge: float = 1.0
    core: float = 1.0
    paired: float = 1.0

    # def total_loss(self, loss: PartLoss) -> Tensor:
    #     return torch.sum([for l in loss])