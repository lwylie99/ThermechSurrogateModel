from dataclasses import dataclass
from typing import Any

import numpy as np

from src.components import NDComponent, PinnSet

@dataclass
class Medium(NDComponent):
    ''' data for a thermal medium '''
    measure: str = ''
    conduction: Any = None
    bcs: PinnSet = None

    def setConditions(self, bounds: PinnSet):
        self.bcs = bounds
        for part in self.axis.fields():
            # sets the outward directions and axis normal to each side
            self.bcs[part].axis = self.axis[part]
            self.bcs[part].direction = self.out[part]
        for part in self.bcs.fields():
            self.bcs[part].k = self.conduction

    def bc(self, part=None):
        if part is None:
            return self.bcs['core']
        return self.bcs[part]


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
