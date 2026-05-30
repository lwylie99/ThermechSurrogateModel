from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.components import NDComponent, PartSet


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
    length: int = None
    width: int = None

    def __init__(self, plate: Medium, units=0.1, x=None, y=None):
        ''' returns sampling grid
         approx 1 interval per x unit of measure + 2
         where +2 is for the boundaries/sides of medium
         '''
        self.length = int(plate.length / units) + 2
        self.width = int(plate.width / units) + 2
        self.grid_map = self.build_grid_map(plate)


    def load_temps(self, filename) -> np.ndarray:
        return self.zeros()

    def zeros(self) -> np.ndarray:
        return np.zeros(self.shape())

    def ones(self) -> np.ndarray:
        return np.ones(self.shape())

    def asMask(self, part):
        return self.masks[part]

    def tensorMask(self, part, grid_map:Tensor=None) -> Tensor:
        ''' a flat boolean mask for use on flattened (N, 2) coord tensor '''
        mask = torch.zeros(self.shape(), dtype=torch.bool)
        mask[self.masks[part]] = True
        mask = mask.reshape(-1)
        if grid_map is not None:
            return grid_map[mask]
        return mask # (rows*cols,)


    def maskSet(self):
        ''' a flat boolean mask for use on flattened (N, 2) coord tensor '''
        masks = PartSet()
        for p in masks.keys(clean=False):
            mask = torch.zeros(self.shape(), dtype=torch.bool)
            mask[self.masks[p]] = True
            masks[p] = mask #.reshape(-1)  # (rows*cols,)
        return masks

    def build_grid_map(self, plate: Medium) -> Tensor:
        ''' MAPS PLATE TO GRID coords[i,j] = [x_cm, y_cm] '''
        xs = torch.linspace(0, plate.length, self.length)
        ys = torch.linspace(0, plate.width, self.width)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # (rows, cols) each
        return torch.stack([xx, yy], dim=-1)  # (rows, cols, 2)

    # def grid_part(self, part='core') -> Tensor:
    #     mask = self.tensorMask(part)
    #     return self.grid_map[mask].detach().requires_grad_(True)

