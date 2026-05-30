from dataclasses import dataclass, asdict

import pandas as pd
import torch
from torch import nn, amp, optim, Tensor
from loss import PartLoss, WeightedLoss
from src import loss
from src.components import Component, PartSet
from src.mediums import Medium, Grid
from src.components_thermal import Gaussian


class BasicMLP(nn.Module):
    ''' MLP with tanh activation and logits as output '''

    def __init__(self, num_in, num_out, num_blocks, num_hidden, dropout=0.2):
        super().__init__()

        layers = []
        l_in, l_out = num_in, num_hidden
        for i in range(num_blocks - 1):
            l_out = num_hidden
            layers += [
                nn.Linear(l_in, l_out),
                # nn.BatchNorm1d(l_out),
                nn.Tanh(),
                nn.Dropout(dropout)
            ]
            l_in = l_out

        layers += [nn.Linear(l_in, num_out)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


@dataclass
class FixedPlateModel(Component):
    '''
    MLP Model for single 2D plate use, only varying factor is the power sources
    num_in: 6
        - predict location (x,y) (conductivity should be shared across plates)
        - gaussian location (x1,y1), amplitude (a), and spread (v)
    num_out: 1
        - temp/stress at (x,y)
    '''
    temp_scale: float = None
    plate: Medium = None
    grid: Grid = None
    grid_map: Tensor = None

    device = None
    model: nn.Module = None
    # scaler: amp.GradScaler = None
    optimizer: optim.Optimizer = None
    lossWts = WeightedLoss()

    # criterion: nn.Module = nn.MSELoss()

    def default_model(self, device='cuda:0'):
        device_str = device if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        # self.scaler = amp.GradScaler(device_str)
        self.model = BasicMLP(6, 1, 6, 256).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)

        self.grid_map = self.grid.build_grid_map(self.plate).reshape(-1, 2).to(self.device)
        #self.grid_map.requires_grad_(True)  # needed for autograd - done on each leaf tensor

    def eval_plate(self, power: Gaussian, ground_truth):
        ''' eval across all points on plate '''
        if self.model is None:
            self.default_model()
        self.model.eval()


    def train_multi_set(self, power: list[Gaussian], epochs=10) -> pd.DataFrame:
        epochs = int(epochs ** (1/2))
        loss = pd.DataFrame()
        for e in range(epochs):
            for p in power:
                p_losses = self.train_plate(p, e//len(power))
                pd.concat([loss, p_losses], ignore_index=True)

        return loss



    def train_plate(self, power: Gaussian, epochs=10, e_start=0) -> pd.DataFrame:
        ''' train across all points on plate '''
        if self.model is None:
            self.default_model()

        self.model.train()

        loss_list = []
        for e in range(epochs):
            self.optimizer.zero_grad()

            loss_parts = dict()
            total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            for part in PartSet().keys(clean=False):
                bc = self.plate.boundaries[part]
                # print((f'\t\tpart: {part}, bc: {bc}').replace("\n", ""))
                if bc is None:
                    # print(f'{part} not found in plate')
                    continue

                # detach from back propagation bc its not fed into model
                part_grid = self.grid.tensorMask(part, self.grid_map).requires_grad_(True)
                # print(f"\tpart: {part.upper()}, part_grid -> shape: {part_grid.shape}, req_grad: {part_grid.requires_grad}, is_leaf: {part_grid.is_leaf}")
                # print(f"\t\tcoords:{part_grid}")
                if part == 'core':
                    bc.build_power_map(part_grid, [power], self.device)

                # with (amp.autocast(self.device.type)):  # allows use of scaler
                temps = self.model_plate(power, coords=part_grid)
                # print(f"\t\ttemps grad_fn: {temps.grad_fn}, temps dtype: {temps.dtype}")

                cur_loss = bc.loss(u=temps, coords=part_grid, k=self.plate.conduction)
                # print(f"\t\tcur_loss: {cur_loss.item():.6f}, requires_grad: {cur_loss.requires_grad}, grad_fn: {cur_loss.grad_fn}")

                part_ratio = (part_grid.shape[0] / self.grid_map.shape[0])
                # print(f'\t\tpart_ratio:{part_ratio}')
                total_loss = total_loss + cur_loss * part_ratio
                loss_parts[part] = cur_loss.item() * part_ratio
                # print(f'\t\tpart: {part}, loss: ', cur_loss)

            # TODO: apply weight
            # total_loss = torch.stack([v for v in loss_parts.items() if v is not None]).sum()
            # self.scaler.scale(total_loss).backward()
            # self.scaler.step(self.optimizer)
            # self.scaler.update()

            total_loss.backward()
            # for name, p in self.model.named_parameters():
            #     if p.grad is None:
            #         print(f"\t\t\tNO GRAD: {name}")
            #     else:
            #         print(f"\t\t\t{name}: grad norm = {p.grad.norm():.6f}")
            self.optimizer.step()

            if e % (epochs // 10) == 0:
                print(f'\tsub_epoch: {e_start+e}, total_loss: ', total_loss)
            loss_parts['total'] = total_loss.item()
            loss_list.append(loss_parts)
            # print(f'\t loss_parts: {loss_parts}')

        return pd.DataFrame(loss_list)


    def model_plate(self, power: Gaussian, coords=None) -> torch.Tensor:
        if coords is None:
            coords = self.grid_map

        n = coords.shape[0]
        gauss_params = torch.tensor(
            [[power.x, power.y, power.amplitude, power.spread]],
            dtype=torch.float32
        ).repeat(n, 1).to(self.device)
        model_input = torch.cat([coords, gauss_params], dim=-1)  # (N, 6)
        preds = self.model(model_input) * self.temp_scale
        return preds
