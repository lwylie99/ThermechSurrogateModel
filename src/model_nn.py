import random
from dataclasses import dataclass
from math import floor
from pathlib import Path

import pandas as pd
import torch
from torch import nn, optim, Tensor

import util_tensor
from loss import LossEngine
from src.components import Component
from src.components_thermal import Gaussian
from src.mediums import Medium, Grid


class FourierFeatures(nn.Module):
    def __init__(self, num_in, num_features, scale=10.0):
        super().__init__()
        B = torch.randn(num_in, num_features // 2) * scale
        self.register_buffer('B', B)

    def forward(self, x):
        proj = x @ self.B  # (N, num_features//2)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, num_features)


class BasicMLP(nn.Module):
    ''' MLP with tanh activation and logits as output '''

    def __init__(self, num_in, num_out, num_blocks, num_hidden, dropout=0.05):
        super().__init__()

        self.layers = []
        l_in, l_out = num_in, num_hidden
        for i in range(num_blocks - 1):
            l_out = num_hidden
            l = nn.Linear(l_in, l_out)
            nn.init.xavier_uniform_(l.weight)
            nn.init.zeros_(l.bias)
            self.layers += [l,
                            nn.Tanh(),
                            nn.Dropout(dropout)
                            ]
            l_in = l_out

        # DO NOT INITIALIZE random wts for last layer
        self.layers += [
            nn.Linear(l_in, num_out),
            nn.Tanh(),
            # nn.Softplus()
        ]  # TODO: new activation function
        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.network(x)

    def save_checkpoint(self, epoch, optimizer, loss, save_dir, check_name=''):
        checkpoint = {
            'epoch': epoch, 'model_state_dict': self.state_dict(),
            'loss': loss, 'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, save_dir / f'checkpoint{check_name}.pth')

    def load_checkpoint(self, optimizer, load_dir, check_name=''):
        load_path = load_dir / f'checkpoint{check_name}.pth'
        print(f'...loading model checkpoint from path: {load_path}')
        checkpoint = torch.load(load_path, weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        return optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def load_model(self, optimizer, path=''):
        checkpoint = torch.load(path, weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        return optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


@dataclass
class ThermalModel2D(Component):
    '''
    MLP Model for single 2D plate use
    '''
    plate: Medium = None
    grid: Grid = None
    grid_map: Tensor = None

    model: BasicMLP = None
    optimizer: optim.Optimizer = None
    engine: LossEngine = None

    checkpoint_dir: Path = None
    _device = None

    temp_scale: float = None
    core_only: bool = False  # if you want to train/eval core without BCs

    def build_model(self, num_in, num_out, num_blocks, num_hidden, lr=1e-3, wt_decay=1e-4, device='cuda'):
        device_str = device if torch.cuda.is_available() else "cpu"
        torch.cuda.empty_cache()
        self._device = torch.device(device_str)
        self.grid_map = self._build_grid_map().to(self._device).requires_grad_(True)

        self.model = BasicMLP(num_in, num_out, num_blocks, num_hidden).to(self._device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wt_decay)

    def set_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def dec_lr(self, divisor):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / divisor

    def save_checkpoint(self, loss, name=''):
        self.model.save_checkpoint(
            epoch=self.engine.e(), optimizer=self.optimizer, loss=loss,
            save_dir=self.checkpoint_dir, check_name=f'_epoch{self.engine.e()}_{name}'
        )

    def load_model(self, load_path=''):
        self.model.load_model(self.optimizer, path=load_path)
        self.engine = LossEngine()

    def load_checkpoint(self, epoch, name='', load_dir=None):
        if load_dir is None:
            load_dir = self.checkpoint_dir
        self.model.load_checkpoint(self.optimizer, load_dir, check_name=f'_epoch{epoch}_{name}')
        self.engine = LossEngine().load_hist(load_dir, epoch=epoch)

    def _build_grid_map(self, plot=False):
        ''' MAPS PLATE TO GRID coords[x,x] = [x_mm, y_mm] '''
        xs, ys, grid_map = util_tensor.build_grid_map(self.plate.shape(), self.grid.shape())
        if plot:
            return xs, ys, grid_map
        return grid_map  # (200, 2)

    def _grid_mask(self, part) -> Tensor:
        ''' a flat boolean mask for use on flattened (N, 2) coord tensor '''
        mask = torch.zeros(self.grid.shape(), dtype=torch.bool)
        mask[self.grid.masks[part]] = True
        mask = mask.reshape(-1)
        return mask

    def _coords(self, part=None) -> Tensor:
        if part is None:
            return self.grid_map
        mask = self._grid_mask(part)
        return self.grid_map[mask]

    def _model(self, model_input: Tensor = None) -> torch.Tensor:
        raw_out = self.model(model_input)
        return raw_out * self.temp_scale


@dataclass
class PowerMapPlateModel(ThermalModel2D):
    '''
    MLP Model for single 2D plate use, only varying factor is the power map
    num_in: 4
        - predict location (x,y) (conductivity should be shared across plates)
        - gaussian power map data at (x,y)
    num_out: 1
        - temp/stress at (x,y)
    '''

    def default_model(self, num_blocks=6, num_hidden=512, lr=1e-3, wt_decay=1e-4, device='cuda'):
        self.build_model(3, 1, num_blocks, num_hidden, lr, wt_decay, device)

    def eval_plate(self, power: list[Gaussian] = None, power_map=None, plot=False):
        with torch.enable_grad():
            self.optimizer.zero_grad()
            power_map, coords, mod_in = self._build_input(power=power, power_map=power_map)
            temps, residuals, total_loss = self._model_plate(coords, mod_in, eval=plot)

            if plot:
                nps = util_tensor.to_numpy([temps, power_map, residuals], self.grid.shape())
                return total_loss, nps[0], nps[1], nps[2]

        return total_loss

    def train_model(self, power_data: list, epochs=24, sub_e=1) -> pd.DataFrame:
        check_int = min(1000, floor(epochs // 3))
        total_epochs = self.engine.e() + epochs
        power_shuffle = power_data.copy()
        running_loss = 0.0
        while self.engine.e() < total_epochs:
            random.shuffle(power_shuffle)
            for p in power_shuffle:
                last_loss = self._train_plate(p, sub_e)
                running_loss += last_loss.item() / check_int
            if  (self.engine.e()-check_int) % check_int == 0:
                print(f'EPOCH[{self.engine.e():5}/{total_epochs:<5}] loss --> '
                      f'avg: {running_loss}, last: ', self.engine.best_loss())
                running_loss = 0.0
                self.save_checkpoint(loss=last_loss)

        print('training complete, saving last checkpoint...')
        self.save_checkpoint(loss=last_loss)
        self.engine.save_hist(self.checkpoint_dir)
        return pd.DataFrame(self.engine.hist)

    def _train_plate(self, power, epochs=1, log_epochs=False):
        ''' train across all points on plate '''
        parts = ['core'] if self.core_only else self.engine.loss_parts()
        total_loss = self.engine.new_epoch().to(self._device)
        for e in range(epochs):
            self.optimizer.zero_grad()
            total_loss = self.engine.new_epoch().to(self._device)
            for part in parts:
                power_map, coords, mod_in = self._build_input([power], part=part)
                temps, cur_loss = self._model_plate(coords, mod_in)
                cur_loss = cur_loss * coords.shape[0]
                total_loss = total_loss + cur_loss
                self.engine.add_part(part, cur_loss)

            total_loss.backward(retain_graph=True)
            self.optimizer.step()
            self.engine.save_epoch(total_loss.item())

            if log_epochs and (self.engine.e() == epochs - 1 or self.engine.e() % (epochs // 2) == 0):
                print(f'\tPWR_EPOCH: {e}, total_loss: ', total_loss)

        return total_loss

    def _build_input(self, power: list[Gaussian] = None, power_map=None, part: str = None) -> tuple[
        Tensor, Tensor, Tensor]:
        coords = self._coords(part).to(self._device)

        if power is not None:
            power_map = self.plate.bcs['core'].build_power_map(coords, power, self._device)
        power_map.to(self._device)

        if coords.shape[0] != power_map.shape[0]:
            print(f"WARNING: coords ({coords.shape}) and power_map ({power_map.shape}) should be same length")

        mod_in = torch.cat(
            [coords.requires_grad_(True), power_map.requires_grad_(True)],
            dim=-1).requires_grad_(True)
        return power_map, coords, mod_in

    def _model_plate(self, coords, model_input, part: str = 'core', eval=False) -> tuple:
        preds = self._model(model_input)
        bc = self.plate.bcs[part]
        cur_loss = bc.loss(u=preds, coords=coords, k=self.plate.conduction)
        if eval:
            residuals = bc.residual(u=preds, coords=coords, k=self.plate.conduction)
            return preds, residuals, cur_loss

        return preds, cur_loss
