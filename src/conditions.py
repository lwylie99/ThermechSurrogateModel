from dataclasses import dataclass

import torch
from torch import Tensor

from src.components import Component, ModularComponent

''' building blocks of a thermal plate set up --> boundary condition parent and power sources '''


@dataclass
class LossComponent(Component):
    comp_type: str = 'LossComponent'

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
        return f'{self.conv_type} at ({self.x}, {self.y})'


# Need to provide power [W] and convert to amplitude [W/m^2]
# For 2D plane: amplitude = power / (2 * pi() * spread^2)
@dataclass
class Gaussian(PowerSource):
    ''' data for a guassian power source '''
    comp_type: str = 'Gaussian'
    internal: bool = True
    amplitude: float = None
    spread: float = None

    def title(self):
        return f'{self.conv_type} at ({self.x}, {self.y}), A={self.amplitude}, σ={self.spread}'
