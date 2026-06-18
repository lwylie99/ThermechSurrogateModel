import util_data
import util_tensor
from util_data import *
from util_plots import *


# def eval_paired_example(model, power_data: list[PMPair], save_dir=None):
#     for i in range(len(power_data)):
#         p = power_data[i]
#         title = p.name
#         total_loss, temps, power_map, residuals = model.eval_plate(p, plot=True)
#         truth_np = p.solution.reshape(model.grid.shape())
#         xs, ys = model._build_grid_map(plot=True)
#         plot_paired_predictions(
#             temps, truth_np, xs, ys,
#             title=title, save_dir=save_dir, save_suffix=f'_comp_{i}'
#         )


def eval_plate_example(model, power_data: list, save_dir=None, inced=None, normal=False):
    for i in range(len(power_data)):
        title = 'Gauss Center Power'
        p = power_data[i]

        xs, ys, grid_map = model._build_grid_map(plot=True)
        if isinstance(p, PMPair) and p.solution is not None:
            print(f'solution shape: {p.solution.reshape(model.grid.shape()).shape}')
            truth_np = util_tensor.to_numpy([p.solution],model.grid.shape())[0]
            norm_range = (np.min(truth_np), np.max(truth_np)) if normal else None
            total_loss, temps, power_map, residuals = model.eval_plate(
                power_map=p.input.to(model._device), plot=True, normal=norm_range
            )
            plot_analytical_comparison(temps, truth_np, xs, ys,
               title=title, save_dir=save_dir, save_suffix=f'_comp_{i}'
            )
        else:
            total_loss, temps, power_map, residuals = model.eval_plate(
                power=p.input, power_map=None, plot=True
            )

        if inced is not None:
            temps = temps + inced

        print(f"shapes --> grid: {model.grid.shape()}, plate: {model.plate.shape()}")
        print(f"       --> power: {power_map.shape}, temps: {temps.shape}, xs: {xs.shape}, ys: {ys.shape}")

        plot_power_map_predictions(temps, power_map, xs, ys,
            title=title, save_dir=save_dir, save_suffix=f'_p{i}'
        )

        plot_pde_residuals(residuals, xs, ys, title=title, save_dir=save_dir, save_suffix=f'_res_{i}')
        # plot_gauss_approx_solution(model, model.plate, model.grid, model.grid_map, p, save_dir=save_dir)

def train_example(model, power_data:list, epochs=100, save_dir=None, compress=None):
    print(f'\nBEGIN TRAINING ({epochs} Epochs)\n')
    loss_hist = model.train_model(power_data=power_data, epochs=epochs)
    plot_example_losshist(trained_model=model, loss_hist=loss_hist, save_hist=True)

def plot_example_losshist(trained_model, loss_hist=None, compress=None, epoch=None, save_dir=None, save_hist=False):
    if loss_hist is None:
        loss_hist = pd.DataFrame(trained_model.engine.hist)

    suffix=''
    if epoch is not None:
        loss_hist = loss_hist.iloc[:epoch]
        suffix=f'{epoch}e'

    if save_dir is not None:
        loss_hist.to_csv(save_dir / f"loss_history{suffix}.csv")

    plot_bc_loss(loss_hist, log_scale=True, compress=compress,
         suffix=suffix, save_dir=save_dir
    )
    plot_total_loss(loss_hist, log_scale=True, suffix=suffix, save_dir=save_dir, compress=compress)
    plot_total_loss(loss_hist, log_scale=False, suffix=suffix, save_dir=save_dir, compress=compress)


def latest_model_path(check_dir: Path) -> str:
    ''' returns the path checkpoint with the largest number of epochs '''
    # model_filename =
    # check_dir = check_dir / model_filename
    print(dir)  # confirm the path
    print(list(check_dir.iterdir()))  # confirm what's in it
    return util_data.last_file(check_dir)