from pathlib import Path

import numpy as np

import data_util
import example_util
from components import PartSet
from power_map_pinns import PowerMapPlateModel
from src.components_thermal import Insulated, Robin, Gaussian, GaussianPde
from src.mediums import Medium, Grid

''' DEFINE PLATE '''
# define FR4 plate
plate = Medium(conduction=0.3, length=40.0, width=40.0)
plate.setConditions(PartSet(
    top=Insulated(),
    bottom=Insulated(),
    left=Robin(h=10.0, ambient=25.0),
    right=Robin(h=10.0, ambient=25.0),
    core=GaussianPde()
))
grid = Grid(plate, units=2.0)
# I added h (convection coefficient) to the definition of Robin so we will have to figure out

''' DEFINE MODEL '''
model_dir = Path(r'checkpoints').resolve()
data_util.clear_dir(model_dir)
model = PowerMapPlateModel(plate=plate, grid=grid, temp_scale=100, model_dir=model_dir)
model.default_model(num_blocks=6, num_hidden=512, lr=0.0001, wt_decay=0.0001, device='cuda:0')

''' TRAIN MODEL '''
train_dir = Path(r'results').resolve()
data_util.clear_dir(train_dir)
fixed_spread, fixed_power = plate.length / 6, 5.0
fixed_amp = fixed_power / (2 * np.pi * fixed_spread ** 2)
power_sources = [
    Gaussian(x=plate.length * 0.5, y=plate.length * 0.5, spread=fixed_spread, amplitude=fixed_amp),
]
example_util.train_example(
    model, power_sources,
    epochs=10, save_dir=train_dir
)

''' EVAL MODEL WITH ANALYTICAL SOLUTION '''
# returns none if empty dir
analytical_pairs = data_util.load_pwrmp_data(Path(r'../../ground_truth').resolve())
eval_data = [analytical_pairs[0]] # should match wih power_sources for initial case
example_util.eval_plate_example(model, eval_data, train_dir)