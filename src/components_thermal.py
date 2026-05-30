from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

from loss import laplacian_jacobian, residual_mse, jacobian
from src.components import Component, ModularComponent


@dataclass
class PowerSource(ModularComponent):
    ''' data for a guassian power source '''
    conv_type: str = 'PowerSource'
    internal: bool = True


@dataclass
class Gaussian(PowerSource):
    ''' data for a guassian power source '''
    conv_type: str = 'Gaussian'
    internal: bool = True
    amplitude: float = None
    spread: float = None


@dataclass
class BoundaryCondition(Component):
    conv_type: str = 'BoundaryCondition'
    def loss(self, u, coords, k) -> Tensor:
        ''' physics loss function '''
        # TODO: does it matter if loss = mse(residual, 0)
        #  vs loss = mse(actual, pred) ?
        return torch.zeros(1)


@dataclass
class PdeCore(BoundaryCondition):
    conv_type: str = 'PdeCore'
    def loss(self, u, coords, k) -> Tensor:
        return torch.zeros(1)


@dataclass
class GaussianPde(PdeCore):
    conv_type: str = 'GaussianPde'
    power_map: Tensor = None

    def build_power_map(self, coords, power: List[Gaussian], device):
        ''' gaussian heat source: Q(x,y) = A * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2)) '''
        amplitude = torch.tensor([g.amplitude for g in power], dtype=torch.float32).to(device)
        spread = torch.tensor([g.spread for g in power], dtype=torch.float32).to(device)
        location = torch.tensor([[g.x, g.y] for g in power], dtype=torch.float32).to(device)

        x, y = coords[:, 0:1], coords[:, 1:2]  # (N, 1)
        x0, y0 = location[..., 0:1], location[..., 1:2]  # (M, 1) or (1,)

        r2 = (x - x0) ** 2 + (y - y0) ** 2  # (N, M) or (N, 1)
        Q = amplitude * torch.exp(-r2 / (2 * spread ** 2))

        self.power_map = Q.sum(dim=-1, keepdim=True)  # (N, 1) — sum over all sources

    def loss(self, u, coords, k) -> Tensor:
        ''' Enforces: k * ∇²u + Q(x,y) = 0 -> residual: k * ∇²u + Q = 0 '''
        jac, u_laplace = laplacian_jacobian(u, coords, k)
        residual = (u_laplace * k + self.power_map).squeeze()
        return residual_mse(residual)

@dataclass
class Edge(BoundaryCondition):
    ''' edges enforce residuals of 0 '''
    conv_type: str = 'Edge'
    axis: int = None
    direction: int = None

@dataclass
class Interface(Edge):
    ''' connection between two blocks, takes 'priority' over other connection types  '''
    conv_type: str = 'Interface'


@dataclass
class Neumann(Edge):
    ''' heat sink '''
    conv_type: str = 'Neumann'
    flux: float = None

    def loss(self, u, coords, k) -> Tensor:
        ''' heat flux (0.0 = adiabatic/insulating)
        Enforces: ∂u/∂n = flux
        '''
        u_jac = jacobian(u, coords)
        residual = (u_jac[..., self.axis] - self.flux).squeeze()
        return residual_mse(residual)


@dataclass
class Insulated(Edge):
    ''' insulated/adiabatic '''
    conv_type: str = 'Insulated'

    def loss(self, u, coords, k) -> Tensor:
        ''' Enforces: ∂u/∂n = 0 (no heat flux through edge) '''
        u_jac = jacobian(u, coords)
        residual = u_jac[..., self.axis].squeeze()
        return residual_mse(residual)


@dataclass
class Robin(Edge):
    ''' convection: ambient/fluid temperature '''
    conv_type: str = 'Robin'
    ambient: float = None

    def loss(self, u, coords, k) -> Tensor:
        ''' convective cooling (Newton's law of cooling)
        Enforces: u - ambient + direction * k * ∂u/∂n = 0
        '''
        u_jac = jacobian(u, coords)
        u_jac = self.direction * k * u_jac[..., self.axis]
        residual = (u.squeeze() - self.ambient + u_jac).squeeze()
        return residual_mse(residual)
