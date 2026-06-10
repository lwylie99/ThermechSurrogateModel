
from util_data import *
from util_plots import *


def eval_paired_example(model, power_data: list[PMPair], save_dir=None):
    for i in range(len(power_data)):
        p = power_data[i]
        title = p.name
        total_loss, temps, power_map, residuals = model.eval_plate(p, plot=True)
        truth_np = p.solution.reshape(model.grid.shape())
        xs, ys = model._build_grid_map(plot=True)
        plot_paired_predictions(
            temps, truth_np, xs, ys,
            title=title, save_dir=save_dir, save_suffix=f'_comp_{i}'
        )


def eval_plate_example(model, power_data: list, save_dir=None):
    for i in range(len(power_data)):
        title = 'Gauss Power'
        p = power_data[i]
        # if isinstance(p, Component):
        #     title = p.title()

        total_loss, temps, power_map, residuals = model.eval_plate(p, plot=True)

        power_map = power_map.cpu().detach().numpy().reshape(model.grid.shape())
        temps = temps.detach().cpu().numpy().reshape(model.grid.shape())
        residuals = residuals.detach().cpu().numpy().reshape(model.grid.shape())
        xs, ys = model._build_grid_map(plot=True)
        print(f"grid  --> len: {model.grid.length}, width: {model.grid.width}")
        print(f"plate --> len: {model.plate.length}, width: {model.plate.width}")
        print(f"shapes --> grid: {model.grid.shape()}, plate: {model.plate.shape()}")
        print(f"       --> power: {power_map.shape}, temps: {temps.shape}")
        print(f"       --> xs: {xs.shape}, ys: {ys.shape}")
        plot_power_map_predictions(
            temps, power_map, xs, ys,
            title=title, save_dir=save_dir, save_suffix=f'_p{i}'
        )

        if hasattr(p, 'solution'):
            truth_np = p.solution.reshape(model.grid.length, model.grid.width)
            plot_paired_predictions(
                temps, truth_np, xs, ys,
                title=title, save_dir=save_dir, save_suffix=f'_comp_{i}'
            )

        plot_pde_residuals(
            residuals, xs, ys, title=title, save_dir=save_dir, save_suffix=f'_res_{i}'
        )

        # plot_gauss_approx_solution(model, model.plate, model.grid, model.grid_map, p, save_dir=save_dir)

def train_example(model, power_data:list, pairs:list=[], epochs=100, save_dir=None):
    print(f'\nBEGIN TRAINING ({epochs} Epochs)\n')
    loss_hist = model.train_model(power_data=power_data, epochs=epochs)
    print(loss_hist)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        loss_hist.to_csv(save_dir / "loss_history.csv")

    # TODO: MAGGIE CONTEXT --> this is where the total_loss_plot and the bc_loss_plot are produced
    plot_bc_loss(loss_hist, log_scale=True, save_dir=save_dir)
    plot_total_loss(loss_hist, log_scale=True, save_dir=save_dir)


