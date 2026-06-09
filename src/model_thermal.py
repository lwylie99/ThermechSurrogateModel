import random
from dataclasses import dataclass

import pandas as pd
import torch

from components_thermal import GaussianPde
from model_nn import ThermalModel2D
from src.components import PartSet
from src.components_thermal import Gaussian


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
            bc = self.plate.boundaries['core']
            coords = self._coords('core')  # ← core only, (144, 2)
            mod_in = self._build_input([power], 'core', bc, coords=coords)
            temps = self._model(mod_in)
            total_loss = bc.loss(u=temps, coords=coords, k=self.plate.conduction)

            if plot:
                # rebuild on full grid for plotting
                full_coords = self._coords()
                bc.build_power_map(full_coords, [power], self.device)
                power_map = bc.power_map
                full_mod_in = self._build_input([power], 'core', bc, coords=full_coords)
                full_temps = self._model(full_mod_in)
                residuals = bc.residual(u=full_temps, coords=full_coords, k=self.plate.conduction)
                return total_loss, full_temps, power_map, residuals

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

            print(f'BEST_LOSS: {best_loss}')

            if e % max(1, (epochs // 10)) == 0:
                print('saving model checkpoint...')
                self.save_checkpoint(e, last_loss)

            if e >= epochs:
                print('training complete, saving last checkpoint...')
                self.save_checkpoint(e, last_loss)
                break

        return pd.DataFrame(loss_hist)

    def _train_plate(self, power, epochs=1):
        ''' train across all points on plate '''
        if self.model is None:
            self.default_model()
        self.model.train()

        parts = PartSet().keys(clean=False)
        if self.core_only:
            parts = ['core']

        loss_list = []
        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for e in range(epochs):
            self.optimizer.zero_grad()

            loss_parts = dict()
            total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            for part in parts:
                bc = self.plate.boundaries[part]
                if bc is None:
                    continue

                # detach from back propagation bc its not fed into model
                coords = self._coords(part)
                mod_in = self._build_input([power], part, bc, coords=coords)
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

        return total_loss.item(), loss_list

    def _build_input(self, power: list[Gaussian], part, bc, coords) -> torch.Tensor:
        if part == 'core':
            bc.build_power_map(coords, power, self.device)
            power_map = bc.power_map
        else:
            temp_bc = GaussianPde()
            temp_bc.build_power_map(coords, power, self.device)
            power_map = temp_bc.power_map

        if coords.shape[0] != power_map.shape[0]:
            print(f"WARNING: coords ({coords.shape}) and power_map ({power_map.shape}) should be same length")

        return torch.cat([
            coords.requires_grad_(True),
            power_map.requires_grad_(True)
        ], dim=-1).requires_grad_(True)











    # sub_scale = min(10, sub_scale * 1.5)
    # sub_e = round(sub_scale)
    #
    # if best_loss < 10:
    #     self.scaler_enabled = False

    # random.shuffle(paired_shuffle)
    # for p in paired_shuffle:
    #     print(f'EPOCH: {e}, pair: ', p.name)
    #     last_loss, p_losses = self._train_pair(p, sub_e)
    #     best_loss = min(best_loss, last_loss)
    #     pair_loss_hist = pair_loss_hist + p_losses
    #     e += sub_e


    # def eval_pair(self, pair: DataPair[np.ndarray], plot=False):
    #     if self.model is None:
    #         self.default_model()
    #     self.model.eval()
    #
    #     with torch.no_grad():
    #         mod_in = self._build_input(power_map=pair.input)
    #         pred_temps = self._model(mod_in)
    #         act_temps = pair.solution
    #         cur_loss = paired_loss(pred_temps, act_temps)
    #         preds_np = pred_temps.detach().cpu().numpy().reshape(self.grid.length, self.grid.width)
    #
    #     xs = torch.linspace(0, self.plate.length, self.grid.length)
    #     ys = torch.linspace(0, self.plate.width, self.grid.width)
    #
    #     if plot:
    #         return cur_loss, preds_np, xs, ys
    #
    #     return cur_loss
    
    


    # def _train_pair(self, pair: PMPair, epochs=10):
    #     ''' train across all points on plate '''
    #     if self.model is None:
    #         self.default_model()
    #     self.model.train()
    #
    #     loss_parts = dict()
    #     loss_list = []
    #     total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
    #     for e in range(epochs):
    #         self.optimizer.zero_grad()
    #         power, act_temps = pair.get_tensors(self.device)
    #         mod_in = self._build_input(power_map=power)
    #         pred_temps = self._model(mod_in)
    #         cur_loss = paired_loss(pred_temps, act_temps)
    #
    #         self._apply_loss(cur_loss)
    #         loss_parts['total'] = cur_loss.item()
    #         loss_list.append(loss_parts)
    #
    #     return total_loss.item(), loss_list
