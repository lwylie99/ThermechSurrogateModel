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
    bcs: PartSet = None

    def setConditions(self, bounds: PartSet):
        self.bcs = bounds
        for part in self.axis.keys():
            # sets the outward directions and axis normal to each side
            self.bcs[part].axis = self.axis[part]
            self.bcs[part].direction = self.out[part]


@dataclass
class Grid(NDComponent):
    ''' sampling grid for a thermal medium '''
    measure: str = 'interval'
    length: int = None
    width: int = None

    def zeros(self) -> np.ndarray:
        return np.zeros(self.shape())

    def ones(self) -> np.ndarray:
        return np.ones(self.shape())

