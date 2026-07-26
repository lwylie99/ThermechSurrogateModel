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

def eval_offset_example(model, power_data: ModelData, offset, title=None, save_dir=None, inced=None, normal=False, suff=''):
    if len(power_data.paired) != len(power_data.pinn):
        print("WARN: all paired data must have associated pinn/power_source data to do offset grid")
        return

    xs, ys, grid_map = model._build_grid_map(offset=offset, plot=True)
    # print(f"shapes --> plate: {model.plate.shape()}, grid: {model.grid.shape()}")
    # print(f"       --> grid_map: {model.grid_map.shape}, xs: {xs.shape}, ys: {ys.shape}")

    for i in range(len(power_data.paired)):
        pinn, pair = power_data._next_pinn(), power_data._next_pair()
        pair.input = None
        offset_data = ModelData(pinn=[pinn], paired=[pair])
        truth_np = util_tensor.to_numpy([pair.solution], model.grid.shape())[0]
        norm_range = (np.min(truth_np), np.max(truth_np)) if normal else None
        total_loss, temps, power_map = model.eval_model(power_data=offset_data, part='paired', normal=norm_range)

        # print(f"       --> p.input: {pair.input.shape}, p.solution: {pair.solution.shape}")
        # print(f"       --> truth_np: {truth_np.shape}, power: {power_map.shape}, temps: {temps.shape}")

        if inced is not None:
            temps = temps + inced

        plot_analytical_comparison(temps, truth_np, xs, ys,
            title=title, save_dir=save_dir, save_suffix=f'_{suff}_offest_evalpair{i}'
        )

        plot_power_map_predictions(temps, power_map, xs, ys,
            title=title, save_dir=save_dir, save_suffix=f'_{suff}_offest_evalpair{i}'
        )

def eval_plate_example(model, power_data: ModelData, title=None, save_dir=None, inced=None, normal=False, suff=''):
    xs, ys, grid_map = model._build_grid_map(plot=True)

    print(f"shapes --> plate: {model.plate.shape()}, grid: {model.grid.shape()}")
    print(f"       --> grid_map: {model.grid_map.shape}, xs: {xs.shape}, ys: {ys.shape}")

    for i in range(len(power_data.paired)):
        p = power_data._next_pair(pop=False)
        truth_np = util_tensor.to_numpy([p.solution], model.grid.shape())[0]
        norm_range = (np.min(truth_np), np.max(truth_np)) if normal else None
        total_loss, temps, power_map = model.eval_model(power_data=power_data, part='paired', normal=norm_range)

        print(f"       --> p.input: {p.input.shape}, p.solution: {p.solution.shape}")
        print(f"       --> truth_np: {truth_np.shape}, power: {power_map.shape}, temps: {temps.shape}")

        if inced is not None:
            temps = temps + inced

        plot_analytical_comparison(temps, truth_np, xs, ys,
            title=title, save_dir=save_dir, save_suffix=f'{suff}_evalpair{i}'
        )

        plot_power_map_predictions(temps, power_map, xs, ys,
            title=title, save_dir=save_dir, save_suffix=f'{suff}_evalpair{i}'
        )

    for i in range(len(power_data.pinn)):
        p = power_data._next_pinn(pop=False)
        total_loss, temps, power_map, residuals = model.eval_model(power_data=power_data)

        if inced is not None:
            temps = temps + inced

        print(f"shapes --> grid: {model.grid.shape()}, plate: {model.plate.shape()}")
        print(f"       --> power: {power_map.shape}, temps: {temps.shape}, xs: {xs.shape}, ys: {ys.shape}")

        plot_power_map_predictions(temps, power_map, xs, ys,
            title=title, save_dir=save_dir, save_suffix=f'_{suff}_p{i}'
        )

        plot_pde_residuals(residuals, xs, ys, title=title, save_dir=save_dir, save_suffix=f'_{suff}_p{i}')
        # plot_gauss_approx_solution(model, model.plate, model.grid, model.grid_map, p, save_dir=save_dir)


def plot_example_losshist(trained_model, loss_hist=None, compress=None, epoch=None, save_dir=None):
    if loss_hist is None:
        loss_hist = pd.DataFrame(trained_model.engine.hist)

    if compress is None:
        compress = max(3, len(loss_hist) // 200)

    suffix = ''
    if epoch is not None:
        loss_hist = loss_hist.iloc[:epoch]
        suffix = f'{epoch}e'

    plot_bc_loss(trained_model, loss_hist, log_scale=True, compress=compress,
        all_parts=False, suffix=suffix, save_dir=save_dir
    )
    plot_bc_loss(trained_model, loss_hist, log_scale=True,
        all_parts=True, suffix=f'_allparts_{suffix}', save_dir=save_dir
    )
    plot_col_loss(loss_hist, col='core', log_scale=True, suffix=suffix, save_dir=save_dir, compress=compress)
    plot_total_loss(loss_hist, log_scale=True, suffix=suffix, save_dir=save_dir, compress=compress)


def latest_model_path(check_dir: Path) -> str:
    ''' returns the path checkpoint with the largest number of epochs '''
    # model_filename =
    # check_dir = check_dir / model_filename
    print(dir)  # confirm the path
    print(list(check_dir.iterdir()))  # confirm what's in it
    return util_data.last_file(check_dir)
