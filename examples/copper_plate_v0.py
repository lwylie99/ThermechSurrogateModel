import json

from components import PartSet
from components_thermal import Neumann
from src.components_thermal import Insulated, Robin, Gaussian, GaussianPde
from src.mediums import Medium
from src.pinns import FixedPlateModel

# define copper plate
# TODO: add weighting to loss
plate = Medium(conduction=0.2, length=1.0, width=1.5)
plate.setConditions(PartSet(
    top=Insulated(),
    bottom=Insulated(),
    left=Robin(ambient=0.2),
    right=Neumann(flux=0.01),
    core=GaussianPde()
))

#

# print('Copper plate top boundary condition type: ', plate.top.transfer_type)
# print('Copper plate bottom boundary condition type: ', plate.bottom.transfer_type)
print('copper plate attribs: ', json.dumps(plate.asDict(), sort_keys=False, indent=4))
print('Copper plate sampling grid (interval per 0.1cm):', plate.asGrid(0.1).asDict())

power_source = Gaussian(x=0.5, y=0.75, spread=0.06, amplitude=1.2)
print('power source: ', power_source.asDict())

grid = plate.asGrid(0.1)
model = FixedPlateModel(plate=plate, grid=grid, temp_scale=300)
model.default_model(device='cuda:0')

# TODO: train
# solution = grid.load_temps('filename')
# model.eval_plate(power_source, solution)
