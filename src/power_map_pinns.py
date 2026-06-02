from dataclasses import dataclass

import pandas as pd
import torch

from data_util import DataPair
from loss import paired_loss
from pinns import ThermalModel2D
from src.components import PartSet
from src.components_thermal import Gaussian


@dataclass
class PowerMapPlateModel(ThermalModel2D):
    '''
    MLP Model for single 2D plate use, only varying factor is the power sources
    num_in: 4
        - predict location (x,y) (conductivity should be shared across plates)
        - gaussian power map data at (x,y)
    num_out: 1
        - temp/stress at (x,y)
    '''

    def default_model(self, num_blocks=6, num_hidden=512, device='cuda:0'):
        self.build_model(4, 1, num_blocks, num_hidden, device)

    def eval_model(self, power: list[Gaussian], plot=True, save_dir=None):
        if self.model is None:
            self.default_model()
        self.model.eval()

    def train_model(self, power_data: list[Gaussian], paired_data: list[DataPair[Gaussian]]=[],epochs=24) -> pd.DataFrame:
        pair_loss_hist = pd.DataFrame()
        last_loss, loss_hist = self._train_plate(power_data[0], 1)
        best_loss = last_loss
        e = 0
        sub_scale = 1
        sub_e = 1
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

            if e % max(1, (epochs // 10)) == 0:
                print('saving model checkpoint...')
                self.save_checkpoint(e, last_loss)

            if e >= epochs:
                print('training complete, saving last checkpoint...')
                self.save_checkpoint(e, last_loss)
                break

            sub_scale = min(20, sub_scale * 1.5)
            sub_e = round(sub_scale)

            if best_loss < 10:
                self.scaler_enabled = False

        return pd.DataFrame(loss_hist)

    def _train_pair(self, pair: DataPair[Gaussian], epochs=10):
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

    def _train_plate(self, power, epochs=10):
        ''' train across all points on plate '''
        if self.model is None:
            self.default_model()
        self.model.train()

        # coords = self._coords('core').requires_grad_(True)
        self.plate.boundaries['core'].build_power_map(self.grid_map, [power], self.device)
        power_map = self.plate.boundaries['core'].power_map()

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
                mod_in = self._build_input(power_map, coords=coords)
                temps = self._model(mod_in)

                cur_loss = bc.loss(u=temps, coords=coords, k=self.plate.conduction)
                # print(f'part:{part}, CUR LOSS: {cur_loss.item()} and MULT: {coords.shape[0]}')
                cur_loss = cur_loss * coords.shape[0]
                total_loss = total_loss + cur_loss
                loss_parts[part] = cur_loss.item()

            self._apply_loss(total_loss)
            loss_parts['total'] = total_loss.item()
            loss_list.append(loss_parts)

            if e == epochs - 1 or e % (epochs // 2) == 0:
                print(f'\tPWR_EPOCH: {e}, total_loss: ', total_loss)

        return total_loss, loss_list

    def _build_input(self, power_map, coords=None) -> torch.Tensor:
        if coords is None:
            coords = self.grid_map

        # coord_features = self.fourier(coords_norm)  # normalized coords into Fourier
        # gauss_params = torch.tensor(power_map, dtype=torch.float32).repeat(coords.shape[0], 1).to(self.device)
        return torch.cat([coords, power_map], dim=-1)

# def _train_plate(self, power: Gaussian, epochs=10, e_start=0) -> pd.DataFrame:
# print(f"\tpart: {part.upper()}, part_grid -> shape: {part_grid.shape}, req_grad: {part_grid.requires_grad}, is_leaf: {part_grid.is_leaf}")
# print(f"\t\tcoords:{part_grid}")
# print(f"\t\ttemps grad_fn: {temps.grad_fn}, temps dtype: {temps.dtype}")
# print(f"\t\tcur_loss: {cur_loss.item():.6f}, requires_grad: {cur_loss.requires_grad}, grad_fn: {cur_loss.grad_fn}")
# print(f'\t loss_parts: {loss_parts}')
