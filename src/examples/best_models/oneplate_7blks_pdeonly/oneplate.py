from pathlib import Path

import numpy as np

import util_data
import util_example
from components import PartSet
from loss import LossEngine
from model_nn import PowerMapPlateModel
from src.components_thermal import Insulated, Robin, Gaussian, GaussianPde
from src.mediums import Medium, Grid

''' DEFINE PLATE & MODEL '''
def get_exp_model_setup():
    plate = Medium(conduction=3e-4, length=40.0, width=40.0) # Update conduction to W/(mm K)
    plate.setConditions(PartSet(
        top=Insulated(),
        bottom=Insulated(),
        left=Robin(h=1e-5, ambient=25.0),
        right=Robin(h=1e-5, ambient=25.0),
        core=GaussianPde()
    ))
    print('PLATE SETUP: ', plate.asJson())

    grid = Grid(length=48, width=48)
    print('GRID SETUP: ', grid)

    wts = PartSet().set(4.0)
    wts.core = 1.0
    loss_engine = LossEngine(loss_wts=wts)
    check_dir = Path(r'checkpoints').resolve()
    model = PowerMapPlateModel(
        plate=plate, grid=grid, engine=loss_engine,
        checkpoint_dir=check_dir, core_only=True
    ) # train on just core to troubleshoot
    model.default_model(
        num_blocks=7, num_hidden=512,
        lr=1e-3, wt_decay=1e-4, device='cuda'
    )
    print('PINN MODEL SETUP: \n', model.model)
    return model

def get_exp_data():
    fixed_spread, fixed_power = 3.0, 0.8
    fixed_amp = fixed_power / (2 * np.pi * fixed_spread ** 2)
    power_sources = [
        Gaussian(x=20.0, y=20.0, spread=fixed_spread, amplitude=fixed_amp),
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
    model.engine.loss_scale = 2**6
    model.core_only = True
    model.set_lr(1e-4)
    print('OPTIMIZER: \n', model.optimizer)

    util_data.clear_dir(model.checkpoint_dir)
    util_data.clear_dir(train_dir)
    util_example.train_example(
        model=model, power_data=power_sources,
        epochs=1000, save_dir=train_dir, compress=5
    )

    print('\nEVAL TRAIN PERFORMANCE... ')
    input_data = [util_data.DataPair(name='CenterGaussianPDE', input=power_sources)]
    util_example.eval_plate_example(model,
        power_data=input_data, save_dir=train_dir #, normal=True #, inced=0,
    )
    util_example.plot_example_losshist(model, compress=4, save_dir=train_dir)
