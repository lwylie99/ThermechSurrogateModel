from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Dict

import pandas as pd
import torch
from torch import Tensor
from torch.autograd import grad

from components import ExpComponent, PinnSet, CompSet


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
class LossSet(CompSet):
    ''' fields exclusive to loss set '''
    paired: Any = None


@dataclass
class PinnLossSet(LossSet, PinnSet):
    ''' fields of both loss set and PINN set '''

    def default_wts(self):
        self.set_all(1.0)


@dataclass
class LossEngine(ExpComponent):
    hist: List[Dict] = None
    load_epoch: int = 0

    part_loss: LossSet = None
    epoch_loss: Tensor = None

    loss_wts: LossSet = None
    loss_scale: float = None

    core_only: bool = False  # if you want to train/eval core without BCs
    paired_freq: int = 0

    def __init__(self, loss_wts:LossSet=None, loss_scale: float=1.0, paired_freq=0):
        ''' paired_freq: paired loss will be applied every x epochs '''
        self.hist = []
        if loss_wts is None:
            loss_wts = PinnLossSet().set_all(1.0)
        self.loss_wts = loss_wts
        self.loss_scale = loss_scale
        self.paired_freq = paired_freq
        self.new_epoch()

    def e(self) -> int:
        ''' returns the current epoch '''
        return len(self.hist)

    def is_checkpoint(self, check_freq) -> bool:
        return check_freq != 0 and (self.e()-self.load_epoch)%check_freq == 0

    def wt(self, part):
        return torch.tensor(self.loss_wts[part], dtype=torch.float32)#.requires_grad_(True)

    def loss_parts(self, as_dict=True):
        ''' returns a LossSet where
            true means it should train against it
            false means it should only eval
        '''
        parts = self.loss_wts.copy(clean=True).set_all(value=(self.core_only==False))
        parts['core'] = self.core_only
        parts['paired'] = self.is_checkpoint(self.paired_freq)
        if not as_dict:
            return parts
        return parts.asDict()

    def best_loss(self):
        func = min  # if mode == 'min' else max
        return func(self.hist, key=lambda x: x['total'])

    def load_hist(self, load_dir: Path, epoch: int = None):
        print(f'...loading loss hist from {load_dir}')
        pre_hist = pd.read_csv(load_dir / 'loss_history.csv')
        pre_hist = pre_hist.loc[:, ~pre_hist.columns.str.startswith('Unnamed:')]
        if epoch is not None:
            pre_hist = pre_hist.iloc[:epoch]
        self.hist = pre_hist.to_dict('records')
        self.load_epoch = self.e()
        return self

    def new_epoch(self):
        self.part_loss = self.loss_wts.copy(values=False, clean=False)
        return torch.tensor(0.0, dtype=torch.float32).requires_grad_(True)

    def add_loss(self, part, loss:Tensor, device):
        if self.part_loss[part] is not None:
            self.part_loss[part] += loss.item()
        else:
            self.part_loss[part] = loss.item()
        loss = loss * self.wt(part).to(device)
        return loss

    def save_epoch(self, loss: Tensor, clean=True):
        loss_dict = self.part_loss.asDict(clean)
        loss_dict['total'] = loss.item()
        loss_dict['epoch'] = self.e() + 1
        self.hist.append(loss_dict)

    def save_hist(self, save_dir: Path):
        pd.DataFrame(self.hist).to_csv(save_dir / "loss_history.csv")
