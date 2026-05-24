from dataclasses import dataclass, asdict

import torch
from torch import nn, amp, optim, Tensor

from components import WeightedLoss
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
                nn.BatchNorm1d(l_out),
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

        self.grid_map = loss.build_grid_map(self.grid, self.plate).reshape(-1, 2).to(self.device)
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

        with (amp.autocast(self.device.type)):  # allows use of scaler
            self.plate.boundaries['core'].build_power_map(power, self.device)
            for e in range(epochs):
                self.optimizer.zero_grad()
                temps = self.model_plate(power)

                loss_parts = PartSet()
                for part in loss_parts.keys():
                    bc = self.plate.boundaries[part]
                    if bc is None:
                        continue

                    mask = self.grid.tensorMask(part)
                    # detach from back propagation bc its not fed into model
                    part_grid = self.grid_map[mask].detach().requires_grad_(True)
                    loss_parts[part] = bc.loss(
                        u=temps[mask], coords=part_grid, k=self.plate.conduction
                    )

                total_loss = sum(v for v in loss_parts.values() if v is not None)

                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

    def model_plate(self, power: Gaussian) -> torch.Tensor:
        n = self.grid_map.shape[0]
        gauss_params = torch.tensor(
            [[power.x, power.y, power.amplitude, power.spread]],
            dtype=torch.float32
        ).repeat(n, 1).to(self.device)
        model_input = torch.cat([self.grid_map, gauss_params], dim=-1)  # (N, 6)

        return self.model(model_input)
