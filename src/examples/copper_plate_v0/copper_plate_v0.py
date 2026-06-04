import json
from pathlib import Path

import data_util
import example_util
from components import PartSet
from src.components_thermal import Insulated, Robin, Gaussian, GaussianPde
from src.mediums import Medium, Grid
from src.pinns import SingleGaussPlateModel

# define copper plate
# TODO: add weighting to loss
plate = Medium(conduction=1.5, length=40, width=40)
plate.setConditions(PartSet(
    top=Insulated(),
    bottom=Insulated(),
    left=Robin(ambient=0.2),
    right=Robin(ambient=0.2),
    core=GaussianPde()
))
grid = Grid(plate, units=1)

model_dir = Path(r'checkpoints').resolve()
data_util.clear_dir(model_dir)
train_dir = Path(r'train_results').resolve()
data_util.clear_dir(train_dir)

model = SingleGaussPlateModel(plate=plate, grid=grid, temp_scale=100, model_dir=model_dir)
model.default_model(device='cuda:0')
fixed_spread, fixed_amp = plate.length/4, 1.0
power_sources = [
    Gaussian(x=plate.length*0.15, y=plate.length*0.15, spread=fixed_spread, amplitude=fixed_amp),
    # Gaussian(x=plate.length*0.15, y=plate.length*0.5, spread=fixed_spread, amplitude=fixed_amp),
    Gaussian(x=plate.length*0.15, y=plate.length*0.85, spread=fixed_spread, amplitude=fixed_amp),
    # Gaussian(x=plate.length*0.5, y=plate.length*0.15, spread=fixed_spread, amplitude=fixed_amp),
    # Gaussian(x=plate.length*0.5, y=plate.length*0.5, spread=fixed_spread, amplitude=fixed_amp),
    # Gaussian(x=plate.length*0.5, y=plate.length*0.85, spread=fixed_spread, amplitude=fixed_amp),
    Gaussian(x=plate.length*0.85, y=plate.length*0.15, spread=fixed_spread, amplitude=fixed_amp),
    # Gaussian(x=plate.length*0.85, y=plate.length*0.5, spread=fixed_spread, amplitude=fixed_amp),
    Gaussian(x=plate.length*0.85, y=plate.length*0.85, spread=fixed_spread, amplitude=fixed_amp),
]

example_util.train_example(model, plate, grid, power_sources, len(power_sources * 100), train_dir)
example_util.eval_plate_example(model, plate, grid, power_sources, train_dir)
#print('copper plate attribs: ', json.dumps(plate.asDict(), sort_keys=False, indent=4))


# eval w new power source
power_source = Gaussian(x=plate.length*0.5, y=plate.length*0.5, spread=fixed_spread, amplitude=fixed_amp)
print('predicting unseen domain -> power source: ', power_source.asDict())
example_util.plot_power_map_predictions(model, plate, grid, model.grid_map, power_source, save_dir=train_dir)