import json
from pathlib import Path

import util_data
import util_example
from components import PartSet
from model_thermal import PowerMapPlateModel
from src.components_thermal import Insulated, Robin, Gaussian, GaussianPde
from src.mediums import Medium, Grid
from src.model_nn import SingleGaussPlateModel

# define copper plate
# TODO: add weighting to loss
plate = Medium(conduction=0.3, length=40.0, width=40.0)
plate.setConditions(PartSet(
    top=Insulated(),
    bottom=Insulated(),
    # for some reason results are flipped from BCs
    left=Robin(ambient=25.0),
    right=Robin(ambient=25.0),
    core=GaussianPde()
))
grid = Grid(plate, units=2.0)

# model_dir = Path(r'./checkpoints').resolve()
# data_util.clear_dir(model_dir)
# train_dir = Path(r'./train_results').resolve()
# data_util.clear_dir(train_dir)

model = PowerMapPlateModel(
    plate=plate, grid=grid, temp_scale=100, model_dir=''
)
model.default_model(
    num_blocks=6, num_hidden=512, lr=0.0001, wt_decay=0.0001, device='cuda'
)
fixed_spread, fixed_amp = plate.length/4, 1.0
power_sources = [
    Gaussian(x=plate.length*0.25, y=plate.length*0.4, spread=fixed_spread, amplitude=fixed_amp)
]

# pairs = data_util.load_pwrmp_data(Path(r'./paired_data').resolve())
# example_util.train_example(
#     model, power_sources, pairs,
#     epochs=1000, save_dir=train_dir
# )
# example_util.eval_plate_example(model, power_sources, train_dir)