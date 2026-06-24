from dataclasses import dataclass
from typing import Any

import numpy as np

from components import PinnSet, EdgeSet, ExpComponent


@dataclass
class ModularComponent(ExpComponent):
    x: float = None
    y: float = None

    def title(self):
        return f'Component at ({self.x}, {self.y})'


@dataclass
class NDComponent(ModularComponent):
    measure: str = None
    length: float | int = None
    width: float | int = None
    masks = PinnSet(
        top=(0, slice(None)),  # y=0, all x   → (1, 20) = 20 pts
        bottom=(-1, slice(None)),  # y=max, all x → 20 pts
        left=(slice(None), 0),  # all y, x=0   → 10 pts
        right=(slice(None), -1),  # all y, x=max → 10 pts
        core=(slice(1, -1), slice(1, -1))  # (8, 18) = 144 pts
    )
    # 0 for x-normal (left/right), 1 for y-normal (top/bottom)
    axis = EdgeSet(top=1, bottom=1, left=0, right=0)
    # direction of normal vector pointing outward
    out = EdgeSet(top=1, bottom=-1, left=-1, right=1)

    def shape(self) -> tuple:
        return self.width, self.length  # (ny, nx) = (rows, cols) — numpy convention

@dataclass
class Medium(NDComponent):
    ''' data for a thermal medium '''
    measure: str = ''
    conduction: float = None
    ambient: float = None
    bcs: PinnSet = None

    def setConditions(self, bounds: PinnSet):
        self.bcs = bounds
        for part in self.axis.fields():
            # sets the outward directions and axis normal to each side
            self.bcs[part].axis = self.axis[part]
            self.bcs[part].direction = self.out[part]
        self.bcs.set_field('k', self.conduction)
        self.bcs.set_field('ambient', self.ambient, replace=False)

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
