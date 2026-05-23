from typing import List

import torch
from torch.autograd import grad

from src.components_mat import Medium, Grid
from src.components_thermal import Gaussian


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


def build_grid_map(grid: Grid, plate: Medium) -> torch.Tensor:
    ''' MAPS PLATE TO GRID coords[i,j] = [x_cm, y_cm] '''
    xs = torch.linspace(0, plate.length, grid.length)
    ys = torch.linspace(0, plate.width, grid.width)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # (rows, cols) each
    return torch.stack([xx, yy], dim=-1)  # (rows, cols, 2)


def gaussian_power_source(coords, power: List[Gaussian], device):
    '''
    Gaussian heat source: Q(x,y) = A * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2))
    coords:    (N, 2) tensor of [x, y] positions
    amplitude: scalar or (M,) tensor for M sources
    spread:    scalar or (M,) tensor, spread of each source
    location:  (2,) or (M, 2) tensor of [x0, y0] for each source
    '''
    amplitude = torch.tensor([g.amplitude for g in power], dtype=torch.float32).to(device)
    spread = torch.tensor([g.spread for g in power], dtype=torch.float32).to(device)
    location = torch.tensor([[g.x, g.y] for g in power], dtype=torch.float32).to(device)

    x, y = coords[:, 0:1], coords[:, 1:2]  # (N, 1)
    x0, y0 = location[..., 0:1], location[..., 1:2]  # (M, 1) or (1,)

    r2 = (x - x0) ** 2 + (y - y0) ** 2  # (N, M) or (N, 1)
    Q = amplitude * torch.exp(-r2 / (2 * spread ** 2))

    return Q.sum(dim=-1, keepdim=True)  # (N, 1) — sum over all sources


def loss_pde_gaussian(u, coords, k, power_map):
    """
    Enforces: k * ∇²u + Q(x,y) = 0  at interior points
    i.e. the residual: k * ∇²u + Q = 0
    """
    jac, u_laplace = laplacian_jacobian(u, coords, k)
    residual = (u_laplace * k + power_map).squeeze()
    return torch.nn.functional.mse_loss(residual, torch.zeros_like(residual))
