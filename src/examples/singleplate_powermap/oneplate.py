from math import floor
from pathlib import Path

import matplotlib
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
    ''' aka get example data '''
    # set up gaussian power source
    fixed_spread, fixed_power = 3.0, 0.8
    fixed_amp = fixed_power / (2 * np.pi * fixed_spread ** 2)
    power_sources = [
        Gaussian(x=20.0, y=20.0, spread=fixed_spread, amp=fixed_amp),
        # Gaussian(x=30.0, y=10.0, spread=fixed_spread, amplitude=fixed_amp),
    ]

    # load analytical/ground truth solutions
    data_path = Path(r'../../ground_truth').resolve()
    analytical_pairs = util_data.load_pwrmp_data(data_path)

    # set up train/truth data for model consumption
    train_data = util_data.ModelData(pinn=[power_sources], paired=analytical_pairs)
    return train_data

def get_exp_results_dir():
    ''' returns results folder within cur directory '''
    return Path(r'results').resolve()



''' DEFINE PLATE & MODEL '''
def get_exp_model_setup():
    # define the thermal medium (aka the heated plate)
    plate = Medium(conduction=3e-4, ambient=25.0,
        measure="mm", length=40.0, width=40.0
    ) # Update conduction to W/(mm K)

    # set the edge/boundary conditions
    plate.setConditions(PinnSet(
        top=Insulated(), bottom=Insulated(),
        left=Robin(h=1e-5), right=Robin(h=1e-5),
        core=GaussianPde()
    ))
    print('PLATE SETUP: ', plate.asJson(clean=False))

    # set up sampling grid
    grid = Grid(length=48, width=48)
    print('GRID SETUP: ', grid)

    # loss engine conducts loss adjustments and records training history
    loss_engine = LossEngine()
    # set the directory used to save model weights
    check_dir = Path(r'checkpoints').resolve()
    # define model wrapper
    model = PowerMapPlateModel(
        plate=plate, grid=grid, engine=loss_engine,
        checkpoint_dir=check_dir
    )
    # define neural network model
    model.default_model(
        num_blocks=5, num_hidden=128,
        lr=1e-3, wt_decay=1e-4,
        device='cuda' # HAIDER - use mps or cpu for mac
    )
    print('PINN MODEL SETUP: \n', model.model)

    # rn this just incs so min is 25
    model.engine.norm_preds = True
    return model

if __name__ == "__main__":
    model = get_exp_model_setup()
    train_data = get_exp_data()
    train_dir = get_exp_results_dir()


    ''' LOAD TRAINED MODEL 
    HAIDER - below are 'rounds' of training, I recommend running one round at a time
    -> comment rounds other than round one, run training
    -> check out results in train_dir
    -> uncomment OPTION 3
    -> comment out rounds other that round 2, run training
    -> repeat for remaining rounds
    -> after training complete, run eval_only.py which will compare prediction to analytical solution
    '''
    # OPTION 1 - use this to load a trained model from a specific place
    # model_path = Path(r'../best_models/best_checkpoints/checkpoint_initialwts_oneplate_pwrmp_epoch100.pth').resolve()
    # model.load_model(model_path)

    # OPTION 2 - use this to load the trained model from an epoch saved in checkpoints dir
    # model.load_checkpoint(epoch=30000)

    # OPTION 3 - load most recent epoch
    # model.load_checkpoint()

    # OPTION 4 - load no model, start from random weights


    ''' TRAIN MODEL '''
    if model.engine.e() == 1 or model.engine.load_epoch == 0:
        util_data.clear_dir(model.checkpoint_dir)
    util_data.clear_dir(train_dir)

    num_epochs = 2000 # usually run training in batches of 1-3k
    model.engine.log_freq = min(1000, floor(num_epochs // 10))
    model.engine.check_freq = min(1000, floor(num_epochs // 2))
    model.engine.eval_freq = 10  # how often prediction is compared to actual solution (not applied to loss)

    print('OPTIMIZER: \n', model.optimizer)


    ''' HAIDER - INITIAL LOSS CONFIG
    loss = (w1*core + w2*side1 + ... + w5*side4 + w6*paired)/6
    w1: (num_core_sample_points^1.7)
    w2-5: (num_sample_points^1.7)
    w6: some very small number, only included every 5th run to avoid 'trumping' the phys-loss
    
    notes
    -> core or pde refers to the gaussian heat source
    -> sides are the boundary conditions
    -> not all loss elements are included in every epoch
    -> if one loss element is below converge_loss it will not be applied to backwards prop
        -> it will still be included in later epochs if above converge_loss
    -> model likes to learn one section at a time, so intermediate results will be lopsided
    '''

    model.engine.core_only = False # best to train with all conditions on from start
    model.engine.paired_freq = 5   # model will ignore phys loss if this is run too frequently
    model.engine.converge_loss = 1e-4
    model.engine.loss_wts.set_wts(
        core=model.grid_map.shape[0]**1.7,
        pinn=float(model.grid_map.shape[0]), paired=1e-4,
    )

    print(f'\nLOSS WTS: {model.engine.loss_wts.asJson()}\n')

    # round 1 - large lr, large converge_loss
    model.set_lr(1e-3)
    loss_hist = model.train_model(train_data=train_data, epochs=num_epochs)
    # plots training progress, most relevant results in loss_core_plot_log, loss_total_plot_log and compare_temps
    util_example.plot_example_losshist(model, loss_hist=loss_hist.copy(), save_dir=train_dir)
    util_example.eval_plate_example(model, power_data=train_data, save_dir=train_dir)

    # round 2 - lower lr
    model.set_lr(1e-4)
    model.engine.converge_loss = 1e-5
    loss_hist = model.train_model(train_data=train_data, epochs=num_epochs)
    util_example.plot_example_losshist(model, loss_hist=loss_hist.copy(), save_dir=train_dir)
    util_example.eval_plate_example(model, power_data=train_data, save_dir=train_dir)

    # round 3 - lower lr, lower converge loss
    model.set_lr(7e-5)
    model.engine.converge_loss = 1e-5
    loss_hist = model.train_model(train_data=train_data, epochs=num_epochs)
    util_example.plot_example_losshist(model, loss_hist=loss_hist.copy(), save_dir=train_dir)
    util_example.eval_plate_example(model, power_data=train_data, save_dir=train_dir)

    # round 3 - lower lr, lower converge loss
    model.set_lr(1e-5)
    model.engine.converge_loss = 1e-7
    loss_hist = model.train_model(train_data=train_data, epochs=num_epochs*2)
    util_example.plot_example_losshist(model, loss_hist=loss_hist.copy(), save_dir=train_dir)
    util_example.eval_plate_example(model, power_data=train_data, save_dir=train_dir)

    # round 3 part 2
    loss_hist = model.train_model(train_data=train_data, epochs=num_epochs*2)
    util_example.plot_example_losshist(model, loss_hist=loss_hist.copy(), save_dir=train_dir)
    util_example.eval_plate_example(model, power_data=train_data, save_dir=train_dir)

    ''' produces extra visuals and graphs and puts them into 'results' dir
    - compare_analytical_norm -> norms predictions to analytical range to evaluate shape
    - compare_analytical_offset (norm or not norm) -> prediction on unseen domain
    '''
    print('\nEVAL TRAIN PERFORMANCE... ')
    util_example.eval_plate_example(model, power_data=train_data, save_dir=train_dir)
