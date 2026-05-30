import json
from pathlib import Path

import plot_util
from components import PartSet
from components_thermal import Neumann
from src.components_thermal import Insulated, Robin, Gaussian, GaussianPde
from src.mediums import Medium, Grid
from src.pinns import FixedPlateModel

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

#

# print('Copper plate top boundary condition type: ', plate.top.transfer_type)
# print('Copper plate bottom boundary condition type: ', plate.bottom.transfer_type)
print('copper plate attribs: ', json.dumps(plate.asDict(), sort_keys=False, indent=4))
# print('Copper plate sampling grid (interval per 0.1cm):', plate.asGrid(0.1).asDict())

grid = Grid(plate, units=0.05)
model = FixedPlateModel(plate=plate, grid=grid, temp_scale=1000)
model.default_model(device='cuda:0')

ps1 = Gaussian(x=1.3, y=0.9, spread=0.3, amplitude=0.8)
print('power source 1: ', ps1.asDict())
ps2 = Gaussian(x=0.1, y=0.3, spread=0.3, amplitude=0.8)
print('power source 2: ', ps2.asDict())

print('\nBEGIN TRAINING\n')
losses = model.train_multi_set([ps1, ps2],1100)
# losses = model.train_plate(power_source,1000)
# print(losses)

save_dir = Path(r'./train_results').resolve()
# plot_util.plot_bc_loss(losses, save_dir=save_dir)
# plot_util.plot_total_loss(losses, save_dir=save_dir)

# TODO: move to eval_plate?
plot_util.plot_predicted_temperature(model, grid, plate, ps1, save_dir=save_dir)
plot_util.plot_temperature_comparison(model, grid, plate, ps1, ambient=10, save_dir=save_dir)

# TODO: move to eval_plate?
plot_util.plot_predicted_temperature(model, grid, plate, ps2, save_dir=save_dir)
plot_util.plot_temperature_comparison(model, grid, plate, ps2, ambient=10, save_dir=save_dir)

# eval w new power source
power_source = Gaussian(x=0.75, y=0.75, spread=0.2, amplitude=0.5)
print('power source: ', power_source.asDict())
plot_util.plot_predicted_temperature(model, grid, plate, power_source, save_dir=save_dir)
plot_util.plot_temperature_comparison(model, grid, plate, power_source, ambient=10, save_dir=save_dir)
# plot_util.plot_power_map(model, grid, plate, power_source, save_dir=save_dir)

