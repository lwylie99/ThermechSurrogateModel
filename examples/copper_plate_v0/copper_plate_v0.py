import json
from pathlib import Path

import plot_util
from components import PartSet
from src.components_thermal import Insulated, Robin, Gaussian, GaussianPde
from src.mediums import Medium, Grid
from src.pinns import SingleGaussPlateModel

# define copper plate
# TODO: add weighting to loss
plate = Medium(conduction=1.1, length=1.5, width=1.5)
plate.setConditions(PartSet(
    top=Insulated(),
    bottom=Insulated(),
    left=Robin(ambient=1.0),
    right=Robin(ambient=1.0),
    core=GaussianPde()
))

print('copper plate attribs: ', json.dumps(plate.asDict(), sort_keys=False, indent=4))

grid = Grid(plate, units=0.05)
model = SingleGaussPlateModel(plate=plate, grid=grid, temp_scale=100)
model.default_model(device='cuda:0')

ps1 = Gaussian(x=1.3, y=0.9, spread=0.3, amplitude=0.65)
print('power source 1: ', ps1.asDict())
ps2 = Gaussian(x=0.75, y=0.75, spread=0.2, amplitude=0.8)
print('power source 2: ', ps2.asDict())

print('\nBEGIN TRAINING\n')
losses = model.train_model([ps1, ps2], 10)

plot_dir = Path(r'./train_results').resolve()

# TODO: move to eval_plate?
plot_util.plot_predictions(model, plate, grid, model.grid_map, ps1, ambient=10, save_dir=plot_dir)

# TODO: move to eval_plate?
plot_util.plot_predictions(model, plate, grid, model.grid_map, ps2, ambient=10, save_dir=plot_dir)

# eval w new power source
power_source = Gaussian(x=0.1, y=0.3, spread=0.5, amplitude=1.0)
print('predicting unseen domain -> power source: ', power_source.asDict())
plot_util.plot_predictions(model, plate, grid, model.grid_map, power_source, ambient=10, save_dir=plot_dir)

