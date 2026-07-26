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
    print(f"checkpoint dir: {model.checkpoint_dir}")
    model.load_checkpoint()

    print('OPTIMIZER: \n', model.optimizer)

    ''' EVAL MODEL '''
    print('\nEVAL PERFORMANCE ON TRAIN DATA... ')
    eval_dir = Path('eval').resolve()
    util_data.clear_dir(eval_dir)
    util_example.eval_plate_example(model, power_data=input_data, normal=True,
        title=f'Center Gauss {model.engine.e()+1} Epochs (Normal)', save_dir=eval_dir
    )
    util_example.plot_example_losshist(model, save_dir=eval_dir)


    print('\nEVAL PERFORMANCE ON OFFSET GRID... ')
    offset = (
        (model.plate.length / model.grid.length) * 0.5,
        (model.plate.width / model.grid.width) * 0.5
    )
    util_example.eval_offset_example(model=model, power_data=input_data, offset=offset,
        title=f'Gauss Power With BCs {model.engine.e()} Epochs (Offset)', save_dir=eval_dir, suff=''
    )
    util_example.eval_offset_example(model=model, power_data=input_data, normal=True, offset=offset,
        title=f'Gauss Power With BCs {model.engine.e()} Epochs (Norm, Off)', save_dir=eval_dir, suff='norm'
    )


    model.load_checkpoint(epoch=29000)
    print(f'...eval pde only training (model at {model.engine.e()}')
    util_example.eval_plate_example(model, power_data=input_data, normal=True,
        title=f'Center Gauss {model.engine.e()} Epochs (Normal)',
        save_dir=eval_dir, suff=f'{model.engine.e()}epochs'
    )

    offset = (
        (model.plate.length / model.grid.length) * 0.5,
        (model.plate.width / model.grid.width) * 0.5
    )
    util_example.eval_offset_example(model=model, power_data=input_data, offset=offset,
        title=f'Gauss Power {model.engine.e()} Epochs (Offset)', save_dir=eval_dir, suff=f'{model.engine.e()}epochs'
    )
    util_example.eval_offset_example(model=model, power_data=input_data, normal=True, offset=offset,
        title=f'Gauss Power {model.engine.e()} Epochs (Norm, Off)', save_dir=eval_dir, suff=f'norm_{model.engine.e()}epochs'
    )


