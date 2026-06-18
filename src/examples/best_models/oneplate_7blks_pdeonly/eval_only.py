from pathlib import Path

import util_data
import util_example
from examples.singleplate_powermap.oneplate import get_exp_model_setup, get_exp_data, get_exp_results_dir

''' LOAD EXP SETUP '''
model = get_exp_model_setup()
power_sources, analytical_pairs = get_exp_data()
train_dir = get_exp_results_dir()

''' LOAD MODEL '''
model.load_checkpoint(epoch=1000)
print('OPTIMIZER: \n', model.optimizer)

''' EVAL MODEL '''
print('\nEVAL PHYS PERFORMANCE... ')
eval_dir = Path('eval').resolve()
input_data = [util_data.DataPair(name='CenterGaussianPDE', input=power_sources)]
util_example.eval_plate_example(model, power_data=input_data, save_dir=train_dir)
util_example.plot_example_losshist(model, compress=5, save_dir=eval_dir)
# util_example.plot_example_losshist(model, compress=3, epoch=1000, save_dir=eval_dir)

print('\nEVAL ANALYTICAL PERFORMANCE... ')
util_example.eval_plate_example(model, power_data=analytical_pairs, save_dir=eval_dir, normal=True) #inced=25,
