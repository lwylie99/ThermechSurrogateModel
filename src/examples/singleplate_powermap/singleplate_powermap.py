from pathlib import Path

import numpy as np

import util_data
import util_example
from components import PartSet
from model_nn import PowerMapPlateModel
from src.components_thermal import Insulated, Robin, Gaussian, GaussianPde
from src.mediums import Medium, Grid

''' DEFINE PLATE '''
# define FR4 plate
plate = Medium(conduction=0.3*1e-3, length=40.0, width=40.0) # Update conduction to W/(mm K)
plate.setConditions(PartSet(
    top=Insulated(),
    bottom=Insulated(),
    left=Robin(h=10.0, ambient=25.0),
    right=Robin(h=10.0, ambient=25.0),
    core=GaussianPde()
))
grid = Grid()
grid.length=48
grid.width=48
# I added h (convection coefficient) to the definition of Robin so we will have to figure out

''' DEFINE MODEL '''
model_dir = Path(r'checkpoints').resolve()
util_data.clear_dir(model_dir)
model = PowerMapPlateModel(
    plate=plate, grid=grid, temp_scale=100,
    model_dir=model_dir, core_only=True
) # train on just core to troubleshoot
model.default_model(num_blocks=6, num_hidden=512, lr=1e-3, wt_decay=1e-4, device='cuda')

''' TRAIN MODEL '''
train_dir = Path(r'results').resolve()
util_data.clear_dir(train_dir)
fixed_spread, fixed_power = 3.0, 0.8
fixed_amp = fixed_power / (2 * np.pi * fixed_spread ** 2)
power_sources = [
    Gaussian(x=20.0, y=20.0, spread=fixed_spread, amplitude=fixed_amp),
]
util_example.train_example(
    model, power_sources,
    epochs=1000, save_dir=train_dir
)
util_example.eval_plate_example(model, power_sources, save_dir=train_dir)

''' EVAL MODEL WITH ANALYTICAL SOLUTION '''
# TODO: mag pls help
# analytical_pairs = util_data.load_pwrmp_data(Path(r'../../ground_truth').resolve()) # returns none if empty dir
# eval_data = [analytical_pairs[0]] # should match wih power_sources for initial case
# util_example.eval_pair_example(model, eval_data, train_dir)