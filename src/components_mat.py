from dataclasses import dataclass, field

import numpy as np
import torch

from src.components import NDComponent, PartSet
from src.components_thermal import Boundary


@dataclass
class MaskSet(PartSet):
    top: tuple = (0, slice(None))
    bottom: tuple = (-1, slice(None))
    left: tuple = (slice(None), 0)
    right: tuple = (slice(None), -1)
    core: tuple = (slice(1, -1), slice(1, -1))


@dataclass
class Grid(NDComponent):
    ''' data for a thermal medium '''
    measure: str = 'interval'
    masks: MaskSet = field(default_factory=MaskSet, repr=False)

    def load_temps(self, filename) -> np.ndarray:
        return self.zeros()

    def zeros(self) -> np.ndarray:
        return np.zeros(self.dims())

    def ones(self) -> np.ndarray:
        return np.ones(self.dims())

    def asMask(self, part):
        return self.masks[part]

    def tensorMask(self, part) -> torch.Tensor:
        ''' a flat boolean mask for use on flattened (N, 2) coord tensor '''
        mask_2d = torch.zeros(self.shape, dtype=torch.bool)
        mask_2d[self[part]] = True
        return mask_2d.reshape(-1)  # (rows*cols,)


@dataclass
class Medium(NDComponent):
    ''' data for a thermal medium '''
    measure: str = 'cm'
    conduction: float = 0.0

    top: Boundary = None
    bottom: Boundary = None
    left: Boundary = None
    right: Boundary = None
    core: Boundary = None

    def asGrid(self, units=0.1) -> Grid:
        ''' returns sampling grid
         approx 1 interval per x unit of measure + 2
         where +2 is for the boundaries/sides of medium
         '''
        return Grid(
            length=self.length // units,
            width=self.width // units,
            # depth = None if self.is2D() else self.depth // units
        )

    def sides(self) -> PartSet:
        return PartSet(self.top, self.bottom, self.left, self.right, self.core)
