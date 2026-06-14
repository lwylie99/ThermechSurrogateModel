from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Dict

import pandas as pd
import torch
from torch import Tensor
from torch.autograd import grad

from components import Component, CompSet, PartSet


def gradients(y, x, create_graph=True, retain_graph=True):
    return grad(y, x,
                grad_outputs=torch.ones_like(y),
                create_graph=create_graph,
                retain_graph=retain_graph
                )


def jacobian(u, coords):
    ''' Inputs: u = temp prediction (N, 1), coords[x, y] = spatial grid (N, 2)
        Outputs: [du/dx, du/dy] = temperature flux (N, 2)
    '''
    return gradients(u, coords)[0]


def laplacian_jacobian(u, coords, k=1):
    ''' k is constant conductivity '''
    jac = jacobian(u, coords)
    uxx = gradients(jac[..., 0] * k, coords)[0][..., 0]
    uyy = gradients(jac[..., 1] * k, coords)[0][..., 1]

    return jac, uxx + uyy


def residual_mse(residual) -> Tensor:
    return torch.nn.functional.mse_loss(residual, torch.zeros_like(residual))


def paired_loss(pred, act) -> Tensor:
    return torch.nn.functional.mse_loss(pred, act)

@dataclass
class LossSet(PartSet):
    ''' to add loss components not included in partset '''
    total: Any = None

    def default_wts(self):
        self.set(1.0)

@dataclass
class LossEngine(Component):
    hist: List[Dict] = None
    wts: CompSet = None
    parts: CompSet = None
    epoch_loss: Tensor = None

    def __init__(self, loss_wts:CompSet=None):
        self.hist = []
        if self.wts is None:
            self.wts = PartSet().set(1.0)
        self.new_epoch()

    def e(self) -> int:
        ''' returns the latest complete epoch '''
        return len(self.hist)

    def loss_parts(self, clean=True):
        return self.wts.fields(clean)

    def best_loss(self):
        func = min #if mode == 'min' else max
        return func(self.hist, key=lambda x: x['total'])

    def load_hist(self, load_dir: Path, epoch: int = None):
        print(f'...loading loss hist from {load_dir}')
        pre_hist = pd.read_csv(load_dir / 'loss_history.csv')
        pre_hist = pre_hist.loc[:, ~pre_hist.columns.str.startswith('Unnamed:')]
        if epoch is not None:
            pre_hist = pre_hist.iloc[:epoch]
        self.hist = pre_hist.to_dict('records')
        return self

    def new_epoch(self):
        loss_type = type(self.wts)
        self.parts = loss_type()
        return torch.tensor(0.0, dtype=torch.float32)

    def add_part(self, part, loss: Tensor):
        self.parts[part] = loss.item()
        return loss

    def save_epoch(self, loss: float, clean=True):
        loss_dict = self.parts.asDict(clean)
        loss_dict['total'] = loss
        self.hist.append(loss_dict)

    def save_hist(self, save_dir: Path):
        pd.DataFrame(self.hist).to_csv(save_dir / "loss_history.csv")
