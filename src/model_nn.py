from dataclasses import dataclass
from dataclasses import dataclass
from math import floor
from pathlib import Path

import pandas as pd
import torch
from torch import nn, optim, Tensor

import util_tensor
from conditions import LossComponent
from conditions_core import PairedData
from loss import LossEngine
from components import ExpComponent
from src.mediums import Medium, Grid
from util_data import ModelData, PowerSourceSet


class FourierFeatures(nn.Module):
    def __init__(self, num_in, num_features, scale=10.0):
        super().__init__()
        B = torch.randn(num_in, num_features // 2) * scale
        self.register_buffer('B', B)

    def forward(self, x):
        proj = x @ self.B  # (N, num_features//2)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, num_features)


class BasicMLP(nn.Module):
    ''' MLP with tanh activation '''

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
                            # nn.Dropout(dropout)
                            ]
            l_in = l_out

        # DO NOT INITIALIZE random wts for last layer
        self.layers += [
            nn.Linear(l_in, num_out),
            # nn.Tanh(),
            nn.Softplus()
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
class ThermalModel2D(ExpComponent):
    '''
    MLP Model for single 2D plate use
    '''
    plate: Medium = None
    grid: Grid = None
    grid_map: Tensor = None

    model: BasicMLP = None
    optimizer: optim.Optimizer = None
    engine: LossEngine = None

    _device: torch.device = None
    checkpoint_dir: Path = None

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

    def load_checkpoint(self, epoch=None, name='', load_dir=None):
        if load_dir is None:
            load_dir = self.checkpoint_dir
        self.engine.load_hist(load_dir, epoch=epoch)
        epoch = self.engine.e()
        self.model.load_checkpoint(self.optimizer, load_dir, check_name=f'_epoch{epoch}_{name}')

    def _build_grid_map(self, offset=(0.0,0.0), plot=False):
        ''' MAPS PLATE TO GRID coords[x,x] = [x_mm, y_mm] '''
        xs, ys, grid_map = util_tensor.build_grid_map(self.plate.shape(), self.grid.shape(), offset)
        if plot:
            return xs, ys, grid_map
        self.grid_map = grid_map
        return self.grid_map

    def _grid_mask(self, part) -> Tensor:
        ''' a flat boolean mask for use on flattened (N, 2) coord tensor '''
        mask = torch.zeros(self.grid.shape(), dtype=torch.bool)
        mask[self.grid.masks[part]] = True
        mask = mask.reshape(-1)
        return mask

    def _coords(self, part=None) -> Tensor:
        if part is None or part == 'paired' or part == 'core':
            return self.grid_map
        mask = self._grid_mask(part)
        return self.grid_map[mask]

    def train_model(self, train_data: ModelData, epochs=24) -> pd.DataFrame:
        print(f"...start epoch: {self.engine.e()}")
        epochs = self.engine.e() + epochs
        while self.engine.e() < epochs and not self.engine.converged:
            last_loss = self._train(train_data)
            if self.engine.is_log_epoch():
                print(f'EPOCH[{self.engine.e():5}/{epochs:<5}] loss --> last:', self.engine.hist[-1])
            if self.engine.is_checkpoint():
                self.save_checkpoint(loss=last_loss)

        print('training complete, saving last checkpoint...')
        self.save_checkpoint(loss=last_loss)
        self.engine.save_hist(self.checkpoint_dir)
        return pd.DataFrame(self.engine.hist)

    def _train(self, power):
        print(f'WARNING: currently in {type(self)} -> _train should be implemented in child class')
        return self.engine.new_epoch().to(self._device)


@dataclass
class PowerMapPlateModel(ThermalModel2D):
    ''' assumes model data is ordered '''

    def default_model(self, num_blocks=6, num_hidden=512, lr=1e-3, wt_decay=1e-4, device='cuda'):
        self.build_model(3, 1, num_blocks, num_hidden, lr, wt_decay, device)

    def _model(self, coords, power_map, bc: LossComponent, eval=False) -> tuple:
        coords = coords.to(self._device)
        power_map = power_map.to(self._device)
        if coords.shape[0] != power_map.shape[0]:
            print(f"WARNING: coords ({coords.shape}) and power_map ({power_map.shape}) should be same length")

        mod_in = torch.cat(
            [coords.requires_grad_(True), power_map.requires_grad_(True)],
        dim=-1).requires_grad_(True).to(self._device)

        preds = self.model(mod_in)
        # normalize for PDE only worked best for training
        if self.engine.norm_preds and coords.shape[0]==self.grid.length*self.grid.width:
            preds = preds - torch.min(preds) + self.plate.ambient
            # preds = util_tensor.normalize(preds.requires_grad_(True),
            #     vmin=self.plate.ambient, vmax=self.plate.ambient+torch.max(preds)-torch.min(preds)
            # ).requires_grad_(True)

        cur_loss = bc.loss(u=preds, coords=coords)
        if eval:
            residuals = bc.residual(u=preds, coords=coords)
            return preds, residuals, cur_loss

        return preds, cur_loss

    def _build_input(self, psource:PowerSourceSet, part) -> tuple[Tensor, Tensor, LossComponent]:
        coords = self._coords(part)
        power_map = self.plate.bc('core').build_power_map(coords, psource.pinn, self._device)
        if part == 'paired':
            psource.paired.input = power_map.to(self._device)
            bc = PairedData(solution=psource.paired.solution.to(self._device))
        else:
            bc = self.plate.bc(part)
        return coords, power_map, bc

    def eval_model(self, power_data: ModelData, part=None, normal:tuple=None):
        with torch.enable_grad():
            self.optimizer.zero_grad()
            power_source = power_data.next()
            coords, power_map, bc = self._build_input(power_source, part=part)
            temps, residuals, total_loss = self._model(coords, power_map, bc, eval=True)

            if normal is not None:
                print(f'...normalizing predictions between {normal}')
                temps = util_tensor.normalize(temps, normal[0], normal[1])

            if part == 'paired':
                nps = util_tensor.to_numpy([temps, power_map], self.grid.shape())
                return total_loss, nps[0], nps[1]

            nps = util_tensor.to_numpy([temps, power_map, residuals], self.grid.shape())
            return total_loss, nps[0], nps[1], nps[2]

    def _train(self, power_data: ModelData):
        ''' PINN training based on gaussian power input '''
        self.optimizer.zero_grad()
        total_loss = self.engine.new_epoch().to(self._device)
        num_parts = 0
        power_source = power_data.next()
        for part, train in self.engine.loss_parts().items():
            if not train and not self.engine.is_eval_epoch():
                continue

            # best PINN results when coords and power map are rebuilt each time
            coords, power_map, bc = self._build_input(power_source, part)
            temps, cur_loss = self._model(coords, power_map, bc, eval=False)
            cur_loss = self.engine.add_loss(part, cur_loss, self._device)

            if train and cur_loss > self.engine.converge_loss:
                # aka, if train is true & loss is large enough, include in total loss
                total_loss = total_loss + cur_loss
                num_parts += 1

        if num_parts == 0:
            self.engine.converged = True
        else:
            total_loss = total_loss / num_parts
            total_loss.backward(retain_graph=True)
            self.optimizer.step()
            self.engine.save_epoch(total_loss)

        return total_loss
