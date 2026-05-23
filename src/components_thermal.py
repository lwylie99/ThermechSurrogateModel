from dataclasses import dataclass, asdict
from typing import Any

from src import loss
from src.components import Component, ModularComponent


@dataclass
class PowerSource(ModularComponent):
    ''' data for a guassian power source '''
    power_type: str = None


@dataclass
class Gaussian(PowerSource):
    ''' data for a guassian power source '''
    power_type: str = 'gaussian'
    spread: float = None
    amplitude: float = None


@dataclass
class Boundary(Component):
    transfer_type: str = None
    loss_function: Any = None


@dataclass
class PdeCore(Boundary):
    transfer_type: str = 'core'


@dataclass
class Interface(Boundary):
    ''' connection between two blocks, takes 'priority' over other connection types  '''
    transfer_type: str = 'interface'


@dataclass
class Neumann(Boundary):
    ''' heat sink '''
    transfer_type: str = 'neumann'


@dataclass
class Insulated(Boundary):
    ''' insulated '''
    transfer_type: str = 'adiabatic'


@dataclass
class Robin(Boundary):
    ''' convection/q_temp: ambient temperature '''
    transfer_type: str = 'robin'
    q_temp: float = 0.0
