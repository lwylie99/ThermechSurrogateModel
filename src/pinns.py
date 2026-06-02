import random
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import torch
from torch import nn, amp, optim, Tensor

from data_util import DataPair
from loss import paired_loss
from src import loss
from src.components import Component, PartSet
from src.mediums import Medium, Grid
from src.components_thermal import Gaussian


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

    def __init__(self, num_in, num_out, num_blocks, num_hidden, dropout=0.01):
        super().__init__()

        layers = []
        l_in, l_out = num_in, num_hidden
        for i in range(num_blocks - 1):
            l_out = num_hidden
            layers += [
                nn.Linear(l_in, l_out),
                nn.Tanh(),
                # nn.Dropout(dropout)
            ]
            l_in = l_out

        layers += [nn.Linear(l_in, num_out)]
        self.network = nn.Sequential(*layers)

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
        checkpoint = torch.load(save_dir/f'checkpoint{check_name}.pth')
        self.load_state_dict(checkpoint['model_state_dict'])
        return optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


@dataclass
class ThermalModel2D(Component):
    '''
    MLP Model for single 2D plate use
    '''
    model_dir: Path = None
    temp_scale: float = None
    plate: Medium = None
    grid: Grid = None
    grid_map: Tensor = None

    device = None
    # fourier: FourierFeatures = None
    model: BasicMLP = None
    optimizer: optim.Optimizer = None
    scaler: amp.GradScaler = None
    scaler_enabled: bool = False

    # criterion: nn.Module = nn.MSELoss()

    def build_model(self, num_in, num_out, num_blocks, num_hidden, device='cuda:0'):
        device_str = device if torch.cuda.is_available() else "cpu"
        torch.cuda.empty_cache()
        self.device = torch.device(device_str)
        self.scaler = amp.GradScaler(device_str)
        self.grid_map = self._build_grid_map().reshape(-1, 2).to(self.device).requires_grad_(True)

        self.model = BasicMLP(num_in, num_out, num_blocks, num_hidden).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(),lr=1e-4, weight_decay=1e-4)
        # self.fourier = FourierFeatures(2, 16, scale=3.0).to(self.device)
        # self.optimizer = optim.Adam(
        #     list(self.model.parameters()) + list(self.fourier.parameters()),
        #     lr=1e-4, weight_decay=1e-4)

    def save_checkpoint(self, epoch, loss, name=''):
        self.model.save_checkpoint(epoch, self.optimizer, loss, self.model_dir, check_name=f'_epoch{epoch}_{name}')

    def load_checkpoint(self, epoch, name=''):
        self.optimizer = self.model.load_checkpoint(self.optimizer, self.model_dir, check_name=f'_epoch{epoch}_{name}')

    def _build_grid_map(self) -> Tensor:
        ''' MAPS PLATE TO GRID coords[i,j] = [x_cm, y_cm] '''
        xs = torch.linspace(0, self.plate.length, self.grid.length)
        ys = torch.linspace(0, self.plate.width, self.grid.width)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # (rows, cols) each
        return torch.stack([xx, yy], dim=-1)  # (rows, cols, 2)

    def _coords(self, part) -> Tensor:
        ''' a flat boolean mask for use on flattened (N, 2) coord tensor '''
        mask = torch.zeros(self.grid.shape(), dtype=torch.bool)
        mask[self.grid.masks[part]] = True
        mask = mask.reshape(-1)
        return self.grid_map[mask]

    def _model(self, model_input: Tensor) -> torch.Tensor:
        if self.scaler_enabled:
            with (amp.autocast(self.device.type)):  # allows use of scaler
                predictions = self.model(model_input) * self.temp_scale
        else:
            predictions = self.model(model_input) * self.temp_scale

        return predictions

    def _apply_loss(self, total_loss: Tensor):
        if self.scaler_enabled:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()



@dataclass
class SingleGaussPlateModel(ThermalModel2D):
    '''
    MLP Model for single 2D plate use, only varying factor is the power sources
    num_in: 6
        - predict location (x,y) (conductivity should be shared across plates)
        - gaussian location (x1,y1), amplitude (a), and spread (v)
    num_out: 1
        - temp/stress at (x,y)
    '''

    def default_model(self, num_blocks=6, num_hidden=256, device='cuda:0'):
        self.build_model(6, 1, 9, 512, device)

    def eval_model(self, power: list[Gaussian], plot=True, save_dir=None):
        if self.model is None:
            self.default_model()
        self.model.eval()


    def train_model(self, power_data:list[Gaussian], paired_data:list[DataPair[Gaussian]]=[], epochs=24) -> pd.DataFrame:
        pair_loss_hist = pd.DataFrame()
        last_loss, loss_hist = self._train_plate(power_data[0], 1)
        best_loss = last_loss
        e=0
        sub_scale=1
        sub_e=1
        power_shuffle = power_data.copy()
        paired_shuffle = paired_data.copy()
        while e < epochs:
            # random.shuffle(power_shuffle)
            for p in power_shuffle:
                print(f'EPOCH: {e}, power: ', p)
                last_loss, p_losses = self._train_plate(p, sub_e)
                best_loss = min(best_loss, last_loss)
                loss_hist = loss_hist + p_losses
                e += sub_e

            # random.shuffle(paired_shuffle)
            for p in paired_shuffle:
                print(f'EPOCH: {e}, pair: ', p)
                last_loss, p_losses = self._train_pair(p, sub_e)
                best_loss = min(best_loss, last_loss)
                pair_loss_hist = pair_loss_hist + p_losses
                e += sub_e

            print(f'BEST_LOSS: {best_loss}')

            if e % max(1, (epochs//10)) == 0:
                print('saving model checkpoint...')
                self.save_checkpoint(e, last_loss)

            if e >= epochs:
                print('training complete, saving last checkpoint...')
                self.save_checkpoint(e, last_loss)
                break

            sub_scale = min(20, sub_scale*1.5)
            sub_e = round(sub_scale)

            if best_loss < 10:
                self.scaler_enabled = False

        return pd.DataFrame(loss_hist)

    def _train_pair(self, pair:DataPair[Gaussian], epochs=10):
        ''' train across all points on plate '''
        if self.model is None:
            self.default_model()
        self.model.train()

        loss_parts = dict()
        loss_list = []
        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for e in range(epochs):
            self.optimizer.zero_grad()
            mod_in = self._build_input(pair.input)
            pred_temps = self._model(mod_in)
            act_temps = pair.solution
            cur_loss = paired_loss(pred_temps, act_temps)

            self._apply_loss(cur_loss)
            loss_parts['total'] = cur_loss.item()
            loss_list.append(loss_parts)

        return total_loss, loss_list

    def _train_plate(self, power: Gaussian, epochs=10):
        ''' train across all points on plate '''
        if self.model is None:
            self.default_model()
        self.model.train()

        loss_list = []
        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for e in range(epochs):
            self.optimizer.zero_grad()

            loss_parts = dict()
            total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            for part in PartSet().keys(clean=False):
                bc = self.plate.boundaries[part]
                if bc is None:
                    continue

                # detach from back propagation bc its not fed into model
                coords = self._coords(part).requires_grad_(True)
                if part == 'core':
                    bc.build_power_map(coords, [power], self.device)

                mod_in = self._build_input(power, coords=coords)
                temps = self._model(mod_in)

                cur_loss = bc.loss(u=temps, coords=coords, k=self.plate.conduction)
                # print(f'part:{part}, CUR LOSS: {cur_loss.item()} and MULT: {coords.shape[0]}')
                cur_loss = cur_loss * coords.shape[0]
                total_loss = total_loss + cur_loss
                loss_parts[part] = cur_loss.item()

            self._apply_loss(total_loss)
            loss_parts['total'] = total_loss.item()
            loss_list.append(loss_parts)

            if e == epochs-1 or e % (epochs // 2) == 0:
                print(f'\tPWR_EPOCH: {e}, total_loss: ', total_loss)

        return total_loss, loss_list

    def _build_input(self, power: Gaussian, coords=None) -> torch.Tensor:
        if coords is None:
            coords = self.grid_map

        # coord_features = self.fourier(coords_norm)  # normalized coords into Fourier
        gauss_params = torch.tensor(
            [[power.x, power.y, power.amplitude, power.spread]],
            dtype=torch.float32
        ).repeat(coords.shape[0], 1).to(self.device)
        return torch.cat([coords, gauss_params], dim=-1)




# def _train_plate(self, power: Gaussian, epochs=10, e_start=0) -> pd.DataFrame:
    # print(f"\tpart: {part.upper()}, part_grid -> shape: {part_grid.shape}, req_grad: {part_grid.requires_grad}, is_leaf: {part_grid.is_leaf}")
    # print(f"\t\tcoords:{part_grid}")
    # print(f"\t\ttemps grad_fn: {temps.grad_fn}, temps dtype: {temps.dtype}")
    # print(f"\t\tcur_loss: {cur_loss.item():.6f}, requires_grad: {cur_loss.requires_grad}, grad_fn: {cur_loss.grad_fn}")
    # print(f'\t loss_parts: {loss_parts}')