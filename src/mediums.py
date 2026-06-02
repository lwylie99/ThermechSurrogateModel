from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from components_thermal import PdeCore
from src.components import NDComponent, PartSet, EdgeSet


@dataclass
class Medium(NDComponent):
    ''' data for a thermal medium '''
    measure: str = ''
    conduction: Any = None
    boundaries: PartSet = None

    def setConditions(self, bounds: PartSet):
        self.boundaries = bounds
        for part in self.axis.keys():
            # sets the outward directions and axis normal to each side
            self.boundaries[part].axis = self.axis[part]
            self.boundaries[part].direction = self.out[part]



@dataclass
class Grid(NDComponent):
    ''' sampling grid for a thermal medium '''
    measure: str = 'interval'
    length: int = None
    width: int = None

    def __init__(self, plate: Medium, units=0.1, x=None, y=None):
        ''' returns sampling grid
         approx 1 interval per x unit of measure + 2
         where +2 is for the boundaries/sides of medium
         '''
        self.length = int(plate.length / units) + 2
        self.width = int(plate.width / units) + 2

    def zeros(self) -> np.ndarray:
        return np.zeros(self.shape())

    def ones(self) -> np.ndarray:
        return np.ones(self.shape())

