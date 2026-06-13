import random
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import torch
from torch import nn, amp, optim, Tensor

import util_torch
from components_thermal import BoundaryCondition
from util_data import DataPair
from loss import paired_loss
from src import loss
from src.components import Component, PartSet
from src.mediums import Medium, Grid
from src.components_thermal import Gaussian, GaussianPde


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
        ] # TODO: new activation function
        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.network(x)

    def save_checkpoint(self, epoch, optimizer, loss, save_dir, check_name=''):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, save_dir/f'checkpoint{check_name}.pth')

    def load_checkpoint(self, optimizer, save_dir, check_name=''):
        checkpoint = torch.load(save_dir/f'checkpoint{check_name}.pth', weights_only=False)
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

    _device = None
    optimizer: optim.Optimizer = None
    model: BasicMLP = None
    model_dir: Path = None

    temp_scale: float = None
    core_only: bool = False    # if you want to train/eval core without BCs

    def build_model(self, num_in, num_out, num_blocks, num_hidden, lr=1e-3, wt_decay=1e-4, device='cuda'):
        device_str = device if torch.cuda.is_available() else "cpu"
        torch.cuda.empty_cache()
        self._device = torch.device(device_str)
        self.grid_map = self._build_grid_map().to(self._device).requires_grad_(True)

        self.model = BasicMLP(num_in, num_out, num_blocks, num_hidden).to(self._device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wt_decay)

    def save_checkpoint(self, epoch, loss, name=''):
        self.model.save_checkpoint(epoch, self.optimizer, loss, self.model_dir, check_name=f'_epoch{epoch}_{name}')

    def load_model(self, path=''):
        self.model.load_model(self.optimizer, path=path)

    def load_checkpoint(self, epoch, name=''):
        self.model.load_checkpoint(self.optimizer, self.model_dir, check_name=f'_epoch{epoch}_{name}')

    def _build_grid_map(self, plot=False):
        ''' MAPS PLATE TO GRID coords[x,x] = [x_mm, y_mm] '''
        xs, ys, grid_map = util_torch.build_grid_map(self.plate.shape(), self.grid.shape())
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

    def _model(self, model_input:Tensor=None) -> torch.Tensor:
        raw_out = self.model(model_input)
        return raw_out * self.temp_scale

    def _apply_loss(self, total_loss: Tensor):
        total_loss.backward(retain_graph=True)
        self.optimizer.step()

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

    def eval_plate(self, power: Gaussian, plot=False):
        with torch.enable_grad():
            self.optimizer.zero_grad()
            power_map, coords, mod_in = self._build_input([power])
            temps, residuals, total_loss = self._model_plate(coords, mod_in, eval=plot)

            if plot:
                nps = util_torch.to_numpy([temps, power_map, residuals], self.grid.shape())
                return total_loss, nps[0], nps[1], nps[2]

        return total_loss


    def train_model(self, power_data: list, epochs=24) -> pd.DataFrame:
        last_loss, loss_hist = float('inf'), []
        best_loss = last_loss
        power_shuffle = power_data.copy()

        e, sub_scale, sub_e = 0, 1, 1
        while e < epochs:
            random.shuffle(power_shuffle)
            for p in power_shuffle:
                print(f'EPOCH: {e}, power: ', p)
                last_loss, p_losses = self._train_plate(p, sub_e)
                best_loss = min(best_loss, last_loss.item())
                loss_hist = loss_hist + p_losses
                e += sub_e

            if e % max(1, (epochs // 10)) == 0:
                print('saving model checkpoint...')
                self.save_checkpoint(e, last_loss)

            if e >= epochs:
                print('training complete, saving last checkpoint...')
                self.save_checkpoint(e, last_loss)
                break

        return pd.DataFrame(loss_hist)

    def _train_plate(self, power, epochs=1, log_epochs=False):
        ''' train across all points on plate '''
        if self.model is None:
            self.default_model()
        self.model.train()

        parts = PartSet().keys()
        if self.core_only:
            parts = ['core']

        loss_list = []
        total_loss = torch.tensor(0.0, device=self._device, dtype=torch.float32)
        for e in range(epochs):
            self.optimizer.zero_grad()

            loss_parts = dict()
            total_loss = torch.tensor(0.0, device=self._device, dtype=torch.float32)
            for part in parts:
                power_map, coords, mod_in = self._build_input([power])
                temps, cur_loss = self._model_plate(coords, mod_in)
                total_loss = total_loss + cur_loss
                loss_parts[part] = cur_loss.item()

            self._apply_loss(total_loss)
            loss_parts['total'] = total_loss.item()
            loss_list.append(loss_parts)

            if log_epochs and (e == epochs - 1 or e % (epochs // 2) == 0):
                print(f'\tPWR_EPOCH: {e}, total_loss: ', total_loss)

        return total_loss, loss_list

    def _build_input(self, power: list[Gaussian], part:str=None) -> tuple[Tensor, Tensor, Tensor]:
        coords = self._coords(part)
        power_map = self.plate.bcs['core'].build_power_map(coords, power, self._device)

        if coords.shape[0] != power_map.shape[0]:
            print(f"WARNING: coords ({coords.shape}) and power_map ({power_map.shape}) should be same length")

        mod_in = torch.cat(
            [coords.requires_grad_(True), power_map.requires_grad_(True)],
        dim=-1).requires_grad_(True)
        return power_map, coords, mod_in

    def _model_plate(self, coords, model_input, part:str='core', eval=False) -> tuple:
        preds = self._model(model_input)
        bc = self.plate.bcs[part]
        cur_loss = bc.loss(u=preds, coords=coords, k=self.plate.conduction)
        cur_loss = cur_loss * coords.shape[0]
        if eval:
            residuals = bc.residual(u=preds, coords=coords, k=self.plate.conduction)
            return preds, residuals, cur_loss

        return preds, cur_loss

# @dataclass
# class SingleGaussPlateModel(ThermalModel2D):
#     '''
#     MLP Model for single 2D plate use, only varying factor is the power sources
#     num_in: 6
#         - predict location (x,y) (conductivity should be shared across plates)
#         - gaussian location (x1,y1), amplitude (a), and spread (v)
#     num_out: 1
#         - temp/stress at (x,y)
#     '''
#
#     def default_model(self, num_blocks=6, num_hidden=256, device='cuda'):
#         self.build_model(6, 1, num_blocks, 512, device)
#
#     def eval_model(self, power: list[Gaussian], plot=True, save_dir=None):
#         if self.model is None:
#             self.default_model()
#         self.model.eval()
#
#
#     def train_model(self, power_data:list[Gaussian], paired_data:list[DataPair[Gaussian]]=[], epochs=24) -> pd.DataFrame:
#         pair_loss_hist = pd.DataFrame()
#         last_loss, loss_hist = self._train_plate(power_data[0], 1)
#         best_loss = last_loss
#         e=0
#         sub_scale=1
#         sub_e=1
#         power_shuffle = power_data.copy()
#         paired_shuffle = paired_data.copy()
#         while e < epochs:
#             # random.shuffle(power_shuffle)
#             for p in power_shuffle:
#                 print(f'EPOCH: {e}, power: ', p)
#                 last_loss, p_losses = self._train_plate(p, sub_e)
#                 best_loss = min(best_loss, last_loss)
#                 loss_hist = loss_hist + p_losses
#                 e += sub_e
#
#             # random.shuffle(paired_shuffle)
#             for p in paired_shuffle:
#                 print(f'EPOCH: {e}, pair: ', p)
#                 last_loss, p_losses = self._train_pair(p, sub_e)
#                 best_loss = min(best_loss, last_loss)
#                 pair_loss_hist = pair_loss_hist + p_losses
#                 e += sub_e
#
#             print(f'BEST_LOSS: {best_loss}')
#
#             if e % max(1, (epochs//10)) == 0:
#                 print('saving model checkpoint...')
#                 self.save_checkpoint(e, last_loss)
#
#             if e >= epochs:
#                 print('training complete, saving last checkpoint...')
#                 self.save_checkpoint(e, last_loss)
#                 break
#
#             sub_scale = min(20, sub_scale*1.5)
#             sub_e = round(sub_scale)
#
#             if best_loss < 10:
#                 self.scaler_enabled = False
#
#         return pd.DataFrame(loss_hist)
#
#     def _train_pair(self, pair:DataPair[Gaussian], epochs=10):
#         ''' train across all points on plate '''
#         if self.model is None:
#             self.default_model()
#         self.model.train()
#
#         loss_parts = dict()
#         loss_list = []
#         total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
#         for e in range(epochs):
#             self.optimizer.zero_grad()
#             mod_in = self._build_input(pair.input)
#             pred_temps = self._model(mod_in)
#             act_temps = pair.solution
#             cur_loss = paired_loss(pred_temps, act_temps)
#
#             self._apply_loss(cur_loss)
#             loss_parts['total'] = cur_loss.item()
#             loss_list.append(loss_parts)
#
#         return total_loss, loss_list
#
#     def _train_plate(self, power: Gaussian, epochs=10):
#         ''' train across all points on plate '''
#         if self.model is None:
#             self.default_model()
#         self.model.train()
#
#         loss_list = []
#         total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
#         for e in range(epochs):
#             self.optimizer.zero_grad()
#
#             loss_parts = dict()
#             total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
#             for part in PartSet().keys(clean=False):
#                 bc = self.plate.boundaries[part]
#                 if bc is None:
#                     continue
#
#                 # detach from back propagation bc its not fed into model
#                 coords = self._coords(part).requires_grad_(True)
#                 if part == 'core':
#                     bc.build_power_map(coords, [power], self.device)
#
#                 mod_in = self._build_input(power, coords=coords)
#                 temps = self._model(mod_in)
#
#                 cur_loss = bc.loss(u=temps, coords=coords, k=self.plate.conduction)
#                 # print(f'part:{part}, CUR LOSS: {cur_loss.item()} and MULT: {coords.shape[0]}')
#                 cur_loss = cur_loss * coords.shape[0]
#                 total_loss = total_loss + cur_loss
#                 total_loss = total_loss
#                 loss_parts[part] = cur_loss.item()
#
#             self._apply_loss(total_loss)
#             loss_parts['total'] = total_loss.item()
#             loss_list.append(loss_parts)
#
#             if e == epochs-1 or e % (epochs // 2) == 0:
#                 print(f'\tPWR_EPOCH: {e}, total_loss: ', total_loss)
#
#         return total_loss, loss_list
#
#     def _build_input(self, power: Gaussian, coords=None) -> torch.Tensor:
#         if coords is None:
#             coords = self.grid_map
#
#         # coord_features = self.fourier(coords_norm)  # normalized coords into Fourier
#         gauss_params = torch.tensor(
#             [[power.x, power.y, power.amplitude, power.spread]],
#             dtype=torch.float32
#         ).repeat(coords.shape[0], 1).to(self.device)
#         return torch.cat([coords, gauss_params], dim=-1)




# def _train_plate(self, power: Gaussian, epochs=10, e_start=0) -> pd.DataFrame:
    # print(f"\tpart: {part.upper()}, part_grid -> shape: {part_grid.shape}, req_grad: {part_grid.requires_grad}, is_leaf: {part_grid.is_leaf}")
    # print(f"\t\tcoords:{part_grid}")
    # print(f"\t\ttemps grad_fn: {temps.grad_fn}, temps dtype: {temps.dtype}")
    # print(f"\t\tcur_loss: {cur_loss.item():.6f}, requires_grad: {cur_loss.requires_grad}, grad_fn: {cur_loss.grad_fn}")
    # print(f'\t loss_parts: {loss_parts}')