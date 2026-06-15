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


def eval_plate_example(trained_model, power_data: list, save_dir=None, inced=None, normal=False):
    for i in range(len(power_data)):
        title = 'Gauss Power'
        p = power_data[i]

        # if isinstance(p, Component):
        #     title = p.title()
        # print(p)
        norm_range = None
        if isinstance(p, PMPair):
            if p.solution is not None:
                truth_np = util_tensor.to_numpy([p.solution.reshape(trained_model.grid.length, trained_model.grid.width)])[0]
                if normal:
                    norm_range = (np.min(truth_np), np.max(truth_np))
            power_map, power = p.input.to(trained_model._device), None
        else:
            power_map, power = None, p.input

        total_loss, temps, power_map, residuals = trained_model.eval_plate(power_map=power_map, power=power, plot=True, normal=norm_range)

        if inced is not None:
            temps = temps + inced

        xs, ys, grid_map = trained_model._build_grid_map(plot=True)
        print(f"shapes --> grid: {trained_model.grid.shape()}, plate: {trained_model.plate.shape()}")
        print(f"       --> power: {power_map.shape}, temps: {temps.shape}, xs: {xs.shape}, ys: {ys.shape}")

        if p.solution is not None:
            plot_paired_predictions(
                temps, truth_np, xs, ys,
                title=title, save_dir=save_dir, save_suffix=f'_comp_{i}'
            )

        plot_power_map_predictions(
            temps, power_map, xs, ys,
            title=title, save_dir=save_dir, save_suffix=f'_p{i}'
        )

        plot_pde_residuals(
            residuals, xs, ys, title=title, save_dir=save_dir, save_suffix=f'_res_{i}'
        )

        # plot_gauss_approx_solution(model, model.plate, model.grid, model.grid_map, p, save_dir=save_dir)

def train_example(model, power_data:list, epochs=100, save_dir=None, compress=None):
    print(f'\nBEGIN TRAINING ({epochs} Epochs)\n')
    loss_hist = model.train_model(power_data=power_data, epochs=epochs)

    if compress is not None:
        loss_hist = compress_dataframe(loss_hist, compress)

    # TODO: MAGGIE CONTEXT --> this is where the total_loss_plot and the bc_loss_plot are produced
    # plot_bc_loss(loss_hist, log_scale=True, save_dir=save_dir)
    plot_total_loss(loss_hist, log_scale=True, save_dir=save_dir)

def latest_model_path(check_dir: Path) -> str:
    ''' returns the path checkpoint with the largest number of epochs '''
    # model_filename =
    # check_dir = check_dir / model_filename
    print(dir)  # confirm the path
    print(list(check_dir.iterdir()))  # confirm what's in it
    return util_data.last_file(check_dir)


def compress_dataframe(df: pd.DataFrame, x: int) -> pd.DataFrame:
    chunk_ids = np.arange(len(df)) // x
    reduced = df.groupby(chunk_ids).mean().reset_index(drop=True)

    # First row index of each chunk
    reduced.insert(0, 'epoch', np.arange(len(reduced)) * x)

    return reduced

def normalize_np(arr, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.min(arr)
    if vmax is None:
        vmax = np.max(arr)
    arr = (arr - vmin) / (vmax - vmin)
    print(arr)
    return arr