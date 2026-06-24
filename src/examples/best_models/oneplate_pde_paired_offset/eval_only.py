from pathlib import Path

import numpy as np

import util_data
import util_example
from conditions import Gaussian
from examples.singleplate_powermap.oneplate import get_exp_model_setup, get_exp_data, get_exp_results_dir

if __name__ == "__main__":
    ''' LOAD EXP SETUP '''
    model = get_exp_model_setup()
    input_data = get_exp_data()
    train_dir = get_exp_results_dir()

    ''' LOAD MODEL '''
    model.load_checkpoint(epoch=1000)

    model.engine.loss_scale = model.grid_map.shape[0]
    model.engine.core_only = True
    model.engine.paired_freq = 5
    model.engine.loss_wts.set_all(1e5)
    model.engine.loss_wts['paired'] = 1e-7
    print('OPTIMIZER: \n', model.optimizer)

    ''' EVAL MODEL '''
    print('\nEVAL PERFORMANCE ON TRAIN DATA... ')
    eval_dir = Path('eval').resolve()
    util_data.clear_dir(eval_dir)
    util_example.eval_plate_example(model, power_data=input_data, normal=True,
        title='Center Gauss PDE & Paired (Normal)', save_dir=eval_dir
    )
    util_example.plot_example_losshist(model, save_dir=eval_dir)


    print('\nEVAL PERFORMANCE ON OFFSET GRID... ')
    pinn_only = util_data.ModelData(pinn=input_data.pinn, paired=[])
    offset = (
        (model.plate.length / model.grid.length) * 0.5,
        (model.plate.width / model.grid.width) * 0.5
    )
    util_example.eval_plate_example(model=model, power_data=pinn_only, offset=offset,
        title='Center Gauss PDE & Paired (Offset)', save_dir=eval_dir, suff='offset'
    )
    util_example.eval_plate_example(model=model, power_data=input_data, normal=True, offset=offset,
        title='Center Gauss PDE & Paired (Normal, Offset)', save_dir=eval_dir, suff='norm_offset'
    )
    util_example.plot_example_losshist(model, save_dir=eval_dir)

    print('\nEVAL NEW DOMAIN... ')
    fixed_spread, fixed_power = 3.0, 0.8
    fixed_amp = fixed_power / (2 * np.pi * fixed_spread ** 2)
    power_sources = [
        Gaussian(x=30.0, y=10.0, spread=fixed_spread, amp=fixed_amp),
    ]
    new_pinn = util_data.ModelData(pinn=[power_sources], paired=[])
    util_example.eval_plate_example(model, power_data=new_pinn, normal=True,
        title='Bottom-Right Gauss PDE & Paired (Unseen)', save_dir=eval_dir, suff='unseen'
    )

    # input_data = util_data.ModelData(pinn=[], paired=analytical_pairs)
    # util_example.eval_plate_example(model, power_data=input_data, title='Center Gauss PDE Only', save_dir=eval_dir, normal=True) #inced=25,
