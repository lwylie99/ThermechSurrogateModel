from pathlib import Path

import util_data
import util_example
from examples.singleplate_powermap.oneplate import get_exp_model_setup, get_exp_data, get_exp_results_dir

if __name__ == "__main__":
    ''' LOAD EXP SETUP '''
    model = get_exp_model_setup()
    input_data = get_exp_data()
    train_dir = get_exp_results_dir()

    ''' LOAD MODEL '''
    model.load_checkpoint(epoch=200)

    model.engine.loss_scale = model.grid_map.shape[0]*1e3
    model.engine.core_only = True
    model.engine.paired_freq = 5
    model.engine.loss_wts['paired'] = 1e-12
    model.set_lr(1e-4)
    print('OPTIMIZER: \n', model.optimizer)

    ''' EVAL MODEL '''
    print('\nEVAL PERFORMANCE... ')
    eval_dir = Path('eval').resolve()
    util_data.clear_dir(eval_dir)
    util_example.eval_plate_example(model, power_data=input_data, title='Center Gauss PDE Only', save_dir=eval_dir, normal=True)
    util_example.plot_example_losshist(model, save_dir=eval_dir)

    # print('\nEVAL ANALYTICAL PERFORMANCE... ')
    # input_data = util_data.ModelData(pinn=[], paired=analytical_pairs)
    # util_example.eval_plate_example(model, power_data=input_data, title='Center Gauss PDE Only', save_dir=eval_dir, normal=True) #inced=25,
