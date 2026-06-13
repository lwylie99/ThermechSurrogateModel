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


def eval_plate_example(trained_model, power_data: list, save_dir=None):
    for i in range(len(power_data)):
        title = 'Gauss Power'
        p = power_data[i]
        # if isinstance(p, Component):
        #     title = p.title()
        print(p)
        if isinstance(p, PMPair):
            total_loss, temps, power_map, residuals = trained_model.eval_plate(power_map=p.input.to(trained_model._device), plot=True)
        else:
            total_loss, temps, power_map, residuals = trained_model.eval_plate(power=p.input, plot=True)

        xs, ys, grid_map = trained_model._build_grid_map(plot=True)
        print(f"shapes --> grid: {trained_model.grid.shape()}, plate: {trained_model.plate.shape()}")
        print(f"       --> power: {power_map.shape}, temps: {temps.shape}, xs: {xs.shape}, ys: {ys.shape}")

        if p.solution is not None:
            truth_np = util_tensor.to_numpy([p.solution.reshape(trained_model.grid.length, trained_model.grid.width)])[0]
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

def train_example(model, power_data:list, epochs=100, save_dir=None):
    print(f'\nBEGIN TRAINING ({epochs} Epochs)\n')
    loss_hist = model.train_model(power_data=power_data, epochs=epochs)
    print(loss_hist)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        loss_hist.to_csv(save_dir / "loss_history.csv")

    # TODO: MAGGIE CONTEXT --> this is where the total_loss_plot and the bc_loss_plot are produced
    plot_bc_loss(loss_hist, log_scale=True, save_dir=save_dir)
    plot_total_loss(loss_hist, log_scale=True, save_dir=save_dir)


