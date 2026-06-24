import dataclasses
from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

from conditions import LossComponent, Gaussian, BoundaryCondition
from loss import laplacian_jacobian, residual_mse, paired_loss

""" for boundary conditions that apply to the code/internal parts of a thermal medium """


@dataclass
class PowerMapCore(BoundaryCondition):
    ''' for data based input/solution loss '''
    comp_type: str = 'PowerMapCore'
    power_map: Tensor = None

    def loss(self, u, coords) -> Tensor:
        ''' u: predicted temperatures, k: material convection (defined in BC parent),
        coords: map of plate locations to sampling grid
        '''
        return torch.zeros(0)


@dataclass
class PairedData(PowerMapCore):
    ''' for data based input/solution loss '''
    comp_type: str = 'PairedData'
    solution: Tensor = None

    def loss(self, u, coords) -> Tensor:
        # print(f"in loss: shapes -> u:{u.shape}, coords:{self.solution.shape}")
        return paired_loss(u, self.solution)


@dataclass
class PdeCore(PowerMapCore):
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
        amplitude, spread, xg, yg = torch.tensor( # order must match return def
            [g.get_values(['amp', 'spread', 'x', 'y']) for g in power],
        dtype=torch.float32, device=device).unbind(dim=-1)

        x, y = coords[:, 0:1], coords[:, 1:2]
        r2 = (x - xg) ** 2 + (y - yg) ** 2
        Q = amplitude * torch.exp(-r2 / (2 * spread ** 2))

        # print("Peak Q =", torch.max(Q), " Amplitude =", amplitude) # Powermap check (should be ~ =)

        self.power_map = Q.sum(dim=-1, keepdim=True).detach()
        return self.power_map

    def residual(self, u, coords) -> Tensor:
        ''' Enforces: k * ∇²u + Q(x,y) = 0
        Returns the raw, un-reduced residual vector for plotting/troubleshooting.
        '''
        # print(f"in residual shapes -> u:{u.shape}, coords:{coords.shape}, k:{self.k}")
        jac, u_laplace = laplacian_jacobian(u, coords, self.k)
        residual = u_laplace + self.power_map.squeeze(-1)
        return residual

    def loss(self, u, coords) -> Tensor:
        ''' Enforces: k * ∇²u + Q(x,y) = 0 -> residual: k * ∇²u + Q = 0 '''
        return residual_mse(self.residual(u, coords))
