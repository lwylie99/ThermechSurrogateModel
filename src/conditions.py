from dataclasses import dataclass

import torch
from torch import Tensor

from mediums import ModularComponent
from components import ExpComponent

''' building blocks of a thermal plate set up --> boundary condition parent and power sources '''


@dataclass
class LossComponent(ExpComponent):
    comp_type: str = 'LossComponent'

    def residual(self, u, coords):
        return torch.zeros(0)

    def loss(self, u, coords) -> Tensor:
        ''' physics loss function '''
        return torch.zeros(0)


@dataclass
class BoundaryCondition(LossComponent):
    comp_type: str = 'BoundaryCondition'
    k: float = None


@dataclass
class PowerSource(ModularComponent):
    ''' data for a guassian power source '''
    comp_type: str = 'PowerSource'
    internal: bool = True

    def title(self):
        return f'{self.comp_type} at ({self.x}, {self.y})'


# Need to provide power [W] and convert to amplitude [W/m^2]
# For 2D plane: amplitude = power / (2 * pi() * spread^2)
@dataclass
class Gaussian(PowerSource):
    ''' data for a guassian power source '''
    comp_type: str = 'Gaussian'
    internal: bool = True
    amp: float = None
    spread: float = None

    def title(self):
        return f'{PowerSource.title(self)}, A={self.amp}, σ={self.spread}'
