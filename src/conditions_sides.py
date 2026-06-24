from dataclasses import dataclass

from torch import Tensor

from conditions import BoundaryCondition
from loss import residual_mse, jacobian

""" for boundary conditions that apply to the edges/sides of a thermal medium """

@dataclass
class Edge(BoundaryCondition):
    ''' edges enforce residuals of 0 '''
    comp_type: str = 'Edge'
    axis: int = None
    direction: int = None


@dataclass
class Interface(Edge):
    ''' connection between two blocks, takes 'priority' over other connection types  '''
    comp_type: str = 'Interface'


@dataclass
class Neumann(Edge):
    ''' heat sink '''
    comp_type: str = 'Neumann'
    flux: float = None

    def loss(self, u, coords) -> Tensor:
        ''' heat flux (0.0 = adiabatic/insulating)
        Enforces: ∂u/∂n = flux
        '''
        u_jac = jacobian(u, coords)
        residual = (u_jac[..., self.axis] - self.flux).squeeze(-1)
        return residual_mse(residual)


@dataclass
class Insulated(Edge):
    ''' insulated/adiabatic '''
    comp_type: str = 'Insulated'

    def loss(self, u, coords) -> Tensor:
        ''' Enforces: ∂u/∂n = 0 (no heat flux through edge) '''
        u_jac = jacobian(u, coords)
        residual = u_jac[..., self.axis].squeeze()
        return residual_mse(residual)


@dataclass
class Robin(Edge):
    ''' convection: ambient/fluid temperature '''
    comp_type: str = 'Robin'
    ambient: float = None
    h: float = None

    def loss(self, u, coords) -> Tensor:
        ''' convective cooling (Newton's law of cooling)
        Enforces: h * (u - ambient) + direction * k * ∂u/∂n = 0
        '''
        u_jac = jacobian(u, coords)
        flux_term = self.direction * self.k
        flux_term = flux_term * u_jac[..., self.axis]
        convection_term = self.h * (u.squeeze() - self.ambient)

        residual = (convection_term + flux_term.squeeze()).squeeze()
        return residual_mse(residual)
