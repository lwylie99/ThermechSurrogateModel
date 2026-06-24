from pathlib import Path

import numpy as np

import util_data
import util_example
from components import PinnSet
from conditions import Gaussian
from conditions_core import GaussianPde
from conditions_sides import Insulated, Robin
from loss import LossEngine
from model_nn import PowerMapPlateModel, PowerMapPlateModel
from src.mediums import Medium, Grid

''' DEFINE PLATE & MODEL '''
def get_exp_model_setup():
    plate = Medium(measure="mm", conduction=3e-4, length=40.0, width=40.0) # Update conduction to W/(mm K)
    plate.setConditions(PinnSet(
        top=Insulated(),
        bottom=Insulated(),
        left=Robin(h=1e-5, ambient=25.0),
        right=Robin(h=1e-5, ambient=25.0),
        core=GaussianPde()
    ))
    print('PLATE SETUP: ', plate.asJson())

    grid = Grid(length=48, width=48)
    print('GRID SETUP: ', grid)

    # wts = PinnSet().set(1.0)
    # wts.core = 10.0
    loss_engine = LossEngine()
    check_dir = Path(r'checkpoints').resolve()
    model = PowerMapPlateModel(
        plate=plate, grid=grid, engine=loss_engine,
        checkpoint_dir=check_dir
    ) # train on just core to troubleshoot
    model.default_model(
        num_blocks=7, num_hidden=512,
        lr=1e-3, wt_decay=1e-4, device='cuda'
    )
    model.engine.loss_scale = 1e2
    model.engine.core_only = True
    model.set_lr(1e-3)
    print('PINN MODEL SETUP: \n', model.model)
    return model

def get_exp_data():
    fixed_spread, fixed_power = 3.0, 0.8
    fixed_amp = fixed_power / (2 * np.pi * fixed_spread ** 2)
    power_sources = [
        Gaussian(x=20.0, y=20.0, spread=fixed_spread, amp=fixed_amp),
    ]

    data_path = Path(r'../../ground_truth').resolve()
    analytical_pairs = util_data.load_pwrmp_data(data_path)
    return power_sources, analytical_pairs

def get_exp_results_dir():
    return Path(r'results').resolve()

if __name__ == "__main__":
    model = get_exp_model_setup()
    power_sources, analytical_pairs = get_exp_data()
    train_dir = get_exp_results_dir()

    ''' TRAIN MODEL '''
    # model_path = Path(r'../best_models/best_checkpoints/checkpoint_initialwts_oneplate_pwrmp_epoch100.pth').resolve()
    # model.load_model(model_path)
    # load_mod_dir = Path(r'../best_models/singleplate_powermap_v3/checkpoints').resolve()
    # model.load_checkpoint(epoch=700, load_dir=load_mod_dir)
    # model.load_checkpoint(epoch=1000)

    ''' when predictions become circle, lr should be set 10 1e-4 
        eventually want to reduce temp scale to 64->32-> ...
    '''
    # is applied before loss calculation to make residuals large enough for loss calculation to be meaningfull
    # by default it is set to grid.length*grid.width, but it should be multiplied by more than 1
    model.engine.loss_scale = 1e2
    model.engine.paired_freq = 0
    model.engine.core_only = True
    model.set_lr(1e-3)
    print('OPTIMIZER: \n', model.optimizer)

    util_data.clear_dir(model.checkpoint_dir)
    util_data.clear_dir(train_dir)

    train_data = util_data.ModelData(pinn=[power_sources], paired=analytical_pairs)
    loss_hist = model.train_model(train_data=train_data, epochs=500)

    print('\nEVAL TRAIN PERFORMANCE... ')
    util_example.eval_plate_example(model,power_data=train_data, save_dir=train_dir)
    util_example.plot_example_losshist(model, loss_hist=loss_hist, save_dir=train_dir)
