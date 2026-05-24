from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.components import NDComponent, PartSet
from src.components_thermal import BoundaryCondition


@dataclass
class Medium(NDComponent):
    ''' data for a thermal medium '''
    measure: str = 'cm'
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

    def __init__(self, plate: Medium, units=0.1, x=None, y=None):
        ''' returns sampling grid
         approx 1 interval per x unit of measure + 2
         where +2 is for the boundaries/sides of medium
         '''
        self.length = plate.length // units
        self.width = plate.width // units


    def load_temps(self, filename) -> np.ndarray:
        return self.zeros()

    def zeros(self) -> np.ndarray:
        return np.zeros(self.shape())

    def ones(self) -> np.ndarray:
        return np.ones(self.shape())

    def asMask(self, part):
        return self.masks[part]

    def tensorMask(self, part) -> torch.Tensor:
        ''' a flat boolean mask for use on flattened (N, 2) coord tensor '''
        mask_2d = torch.zeros(self.shape(), dtype=torch.bool)
        mask_2d[self.masks[part]] = True
        return mask_2d.reshape(-1)  # (rows*cols,)

    def build_grid_map(self, plate: Medium) -> torch.Tensor:
        ''' MAPS PLATE TO GRID coords[i,j] = [x_cm, y_cm] '''
        xs = torch.linspace(0, plate.length, self.length)
        ys = torch.linspace(0, plate.width, self.width)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # (rows, cols) each
        return torch.stack([xx, yy], dim=-1)  # (rows, cols, 2)