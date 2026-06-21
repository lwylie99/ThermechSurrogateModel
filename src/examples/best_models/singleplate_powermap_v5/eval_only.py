from pathlib import Path

import util_data
import util_example
from examples.singleplate_powermap.oneplate import get_exp_model_setup, get_exp_data, get_exp_results_dir

if __name__ == "__main__":
    ''' LOAD EXP SETUP '''
    model = get_exp_model_setup()
    power_sources, analytical_pairs = get_exp_data()
    train_dir = get_exp_results_dir()

    ''' LOAD MODEL '''
    model.load_checkpoint(epoch=500)
    print('OPTIMIZER: \n', model.optimizer)

    ''' EVAL MODEL '''
    print('\nEVAL PHYS PERFORMANCE... ')
    eval_dir = Path('eval').resolve()
    util_data.clear_dir(eval_dir)
    input_data = util_data.ModelData(pinn=[power_sources], paired=[])
    util_example.eval_plate_example(model, power_data=input_data, title='Center Gauss PDE Only', save_dir=eval_dir)
    util_example.plot_example_losshist(model, save_dir=eval_dir)

    print('\nEVAL ANALYTICAL PERFORMANCE... ')
    input_data = util_data.ModelData(pinn=[], paired=analytical_pairs)
    util_example.eval_plate_example(model, power_data=input_data, title='Center Gauss PDE Only', save_dir=eval_dir, normal=True) #inced=25,
