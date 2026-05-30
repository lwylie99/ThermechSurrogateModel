from dataclasses import dataclass, asdict

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
                nn.ReLU(),
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
    scaler: amp.GradScaler = None
    optimizer: optim.Optimizer = None
    lossWts = WeightedLoss()

    # criterion: nn.Module = nn.MSELoss()

    def default_model(self, device='cuda:0'):
        device_str = device if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        self.scaler = amp.GradScaler(device_str)
        self.model = BasicMLP(6, 1, 5, 128).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)

        self.grid_map = self.grid.build_grid_map(self.plate).reshape(-1, 2).to(self.device)
        self.grid_map.requires_grad_(True)  # needed for autograd

    def eval_plate(self, power: Gaussian, ground_truth):
        ''' eval across all points on plate '''
        if self.model is None:
            self.default_model()
        self.model.eval()

    def train_plate(self, power: Gaussian, epochs=1):
        ''' train across all points on plate '''
        if self.model is None:
            self.default_model()

        self.model.train()

        # build power map and masks before training
        part_grid = self.grid.tensorMask('core', self.grid_map).detach().requires_grad_(True)
        self.plate.boundaries['core'].build_power_map(part_grid, [power], self.device)

        with (amp.autocast(self.device.type)):  # allows use of scaler
            for e in range(epochs):
                print(f'sub_epoch: {e}')
                self.optimizer.zero_grad()

                loss_parts = PartLoss()
                for part in loss_parts.keys(clean=False):
                    bc = self.plate.boundaries[part]
                    print(f'\t\tpart: {part}, bc: {bc}')
                    if bc is None:
                        print(f'{part} not found in plate')
                        continue

                    # detach from back propagation bc its not fed into model
                    part_grid = self.grid.tensorMask(part, self.grid_map).detach().requires_grad_(True)
                    temps = self.model_plate(power, coords=part_grid)
                    loss_parts[part] = bc.loss(
                        u=temps, coords=part_grid, k=self.plate.conduction
                    )

                # TODO: apply weight
                total_loss = torch.stack([v for v in loss_parts.values() if v is not None]).sum()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                print(f'\t loss_parts: {loss_parts}')


    def model_plate(self, power: Gaussian, coords=None) -> torch.Tensor:
        if coords is None:
            coords = self.grid_map

        n = coords.shape[0]
        gauss_params = torch.tensor(
            [[power.x, power.y, power.amplitude, power.spread]],
            dtype=torch.float32
        ).repeat(n, 1).to(self.device)
        model_input = torch.cat([coords, gauss_params], dim=-1)  # (N, 6)

        return self.model(model_input)
