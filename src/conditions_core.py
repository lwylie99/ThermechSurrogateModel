from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

from conditions import LossComponent, Gaussian, BoundaryCondition
from loss import laplacian_jacobian, residual_mse, paired_loss

""" for boundary conditions that apply to the code/internal parts of a thermal medium """


@dataclass
class PowerMapCore(LossComponent):
    ''' for data based input/solution loss '''
    comp_type: str = 'PowerMapCore'
    power_map: Tensor = None

    def loss(self, u, coords) -> Tensor:
        ''' u: predicted temperatures, k: material convection,
        coords: map of plate locations to sampling grid
        '''
        return torch.zeros(0)


@dataclass
class PairedData(PowerMapCore):
    ''' for data based input/solution loss '''
    comp_type: str = 'PairedData'
    solution: Tensor = None

    def loss(self, u, coords) -> Tensor:
        return paired_loss(u, self.solution)


@dataclass
class PdeCore(BoundaryCondition, PowerMapCore):
    ''' for PDE based loss conditions applied to whole map '''
    comp_type: str = 'PdeCore'

    def build_power_map(self, coords, power, device) -> Tensor:
        return torch.zeros(1)


@dataclass
class GaussianPde(PdeCore):
    comp_type: str = 'GaussianPde'
    power_map: Tensor = None

    # amplitude = power / (2 * pi() * spread^2)

    def build_power_map(self, coords, power: List[Gaussian], device):
        ''' gaussian heat source: Q(x,y) = A * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2)) '''
        amplitude = torch.tensor([g.amplitude for g in power], dtype=torch.float32).to(device)
        spread = torch.tensor([g.spread for g in power], dtype=torch.float32).to(device)
        location = torch.tensor([[g.x, g.y] for g in power], dtype=torch.float32).to(device)

        x, y = coords[:, 0:1], coords[:, 1:2]  # (N, 1)
        x0, y0 = location[:, 0], location[:, 1]  # (M, 1) or (1,)
        r2 = (x - x0) ** 2 + (y - y0) ** 2  # (N, M) or (N, 1)
        # amplitude
        Q = amplitude * torch.exp(-r2 / (2 * spread ** 2))

        # Powermap check (should be ~ =)
        # print("Peak Q =", torch.max(Q), " Amplitude =", amplitude)

        self.power_map = Q.sum(dim=-1, keepdim=True).detach()  # (N, 1) — sum over all sources
        return self.power_map

    def residual(self, u, coords) -> Tensor:
        ''' Enforces: k * ∇²u + Q(x,y) = 0
        Returns the raw, un-reduced residual vector for plotting/troubleshooting.
        '''
        jac, u_laplace = laplacian_jacobian(u, coords, self.k)
        residual = u_laplace + self.power_map.squeeze(-1)
        return residual

    def loss(self, u, coords) -> Tensor:
        ''' Enforces: k * ∇²u + Q(x,y) = 0 -> residual: k * ∇²u + Q = 0 '''
        return residual_mse(self.residual(u, coords))
