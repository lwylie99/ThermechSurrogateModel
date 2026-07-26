from math import floor
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

def get_exp_data() -> util_data.ModelData:
    fixed_spread, fixed_power = 3.0, 0.8
    fixed_amp = fixed_power / (2 * np.pi * fixed_spread ** 2)
    power_sources = [
        Gaussian(x=20.0, y=20.0, spread=fixed_spread, amp=fixed_amp),
        # Gaussian(x=30.0, y=10.0, spread=fixed_spread, amplitude=fixed_amp),
    ]

    data_path = Path(r'../../ground_truth').resolve()
    analytical_pairs = util_data.load_pwrmp_data(data_path)
    train_data = util_data.ModelData(pinn=[power_sources], paired=analytical_pairs)
    return train_data

def get_exp_results_dir():
    return Path(r'results').resolve()



''' DEFINE PLATE & MODEL '''
def get_exp_model_setup():
    plate = Medium(conduction=3e-4, ambient=25.0,
        measure="mm", length=40.0, width=40.0
    ) # Update conduction to W/(mm K)
    plate.setConditions(PinnSet(
        top=Insulated(), bottom=Insulated(),
        left=Robin(h=1e-5), right=Robin(h=1e-5),
        core=GaussianPde()
    ))
    print('PLATE SETUP: ', plate.asJson(clean=False))

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
        num_blocks=5, num_hidden=128,
        lr=1e-3, wt_decay=1e-4, device='cuda'
    )
    print('PINN MODEL SETUP: \n', model.model)

    # rn this just incs so min is 25
    model.engine.norm_preds = True
    return model

if __name__ == "__main__":
    model = get_exp_model_setup()
    train_data = get_exp_data()
    train_dir = get_exp_results_dir()

    ''' TRAIN MODEL '''
    # model_path = Path(r'../best_models/best_checkpoints/checkpoint_initialwts_oneplate_pwrmp_epoch100.pth').resolve()
    # model.load_model(model_path)
    # load_mod_dir = Path(r'../best_models/singleplate_powermap_v3/checkpoints').resolve()
    # model.load_checkpoint(epoch=700, load_dir=load_mod_dir)
    model.load_checkpoint(epoch=30000) # first 1k were paired and PDE

    # if model.engine.e() == 1 or model.engine.load_epoch == 0:
    # util_data.clear_dir(model.checkpoint_dir)
    util_data.clear_dir(train_dir)

    num_epochs = 20000
    model.engine.log_freq = min(1000, floor(num_epochs // 10))
    model.engine.check_freq = min(1000, floor(num_epochs // 2))
    model.engine.eval_freq = 10

    model.engine.paired_freq = 5
    # model.engine.loss_wts.set_wts(
    #     core=model.grid_map.shape[0]**1.7,
    #     pinn=float(model.grid_map.shape[0]), paired=1e-4,
    # )
    model.engine.loss_wts.set_wts(
        core=float(model.grid_map.shape[0]**1.7),
        pinn=float(model.grid_map.shape[0]/5), paired=1e-4
    )
    # model.engine.loss_wts['top'] = 1
    print('OPTIMIZER: \n', model.optimizer)
    print(f'\nLOSS WTS: {model.engine.loss_wts.asJson()}\n')

    ''' start with PDE/PAIRED only'''
    model.engine.core_only = False
    # model.engine.converge_loss = 1e-7
    # model.set_lr(1e-3)
    # loss_hist = model.train_model(train_data=train_data, epochs=4000)
    # util_example.plot_example_losshist(model, loss_hist=loss_hist.copy(), save_dir=train_dir)

    # model.set_lr(1e-4)
    # model.engine.converge_loss = 1e-7
    # loss_hist = model.train_model(train_data=train_data, epochs=3000)
    # util_example.plot_example_losshist(model, loss_hist=loss_hist.copy(), save_dir=train_dir)

    # model.set_lr(7e-5)
    # model.engine.converge_loss = 5e-7
    # loss_hist = model.train_model(train_data=train_data, epochs=3000)
    # util_example.plot_example_losshist(model, loss_hist=loss_hist.copy(), save_dir=train_dir)

    model.set_lr(1e-5)
    model.engine.converge_loss = 4e-7
    loss_hist = model.train_model(train_data=train_data, epochs=10000)
    util_example.plot_example_losshist(model, loss_hist=loss_hist.copy(), save_dir=train_dir)


    print('\nEVAL TRAIN PERFORMANCE... ')
    util_example.eval_plate_example(model, power_data=train_data, save_dir=train_dir)
