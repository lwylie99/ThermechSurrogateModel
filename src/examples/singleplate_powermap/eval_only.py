import util_data
import util_example
from examples.singleplate_powermap.singleplate_powermap import get_exp_model_setup, get_exp_data, get_exp_results_dir

''' LOAD EXP SETUP '''
model = get_exp_model_setup()
power_sources, analytical_pairs = get_exp_data()
train_dir = get_exp_results_dir()

''' LOAD MODEL '''
model.load_checkpoint(epoch=9000)
print('OPTIMIZER: \n', model.optimizer)

''' EVAL MODEL '''
print('\nEVAL PHYS PERFORMANCE... ')
input_data = [util_data.DataPair(name='CenterGaussianPDE', input=power_sources)]
util_example.eval_plate_example(model, power_data=input_data, save_dir=train_dir)

print('\nEVAL ANALYTICAL PERFORMANCE... ')
util_example.eval_plate_example(model, analytical_pairs, train_dir)
