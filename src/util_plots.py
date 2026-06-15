import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from components import Component
from components_thermal import GaussianPde
from util_data import DataPair, PMPair

def plot_bc_loss(df: pd.DataFrame, log_scale=False, save_dir=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in df.columns:
        if col == 'total':
            continue
        ax.plot(df.index, df[col].dropna(), label=col, linewidth=0.5, linestyle='-')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("BC Loss")
    ax.legend()
    if log_scale:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "bc_loss_plot.png", dpi=150)
        df.to_csv(save_dir / "loss_history.csv")
        print(f"Saved to {save_dir}")

    #plt.show()


def plot_total_loss(df: pd.DataFrame, log_scale=False, save_dir=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    if 'epoch' in df:
        ax.plot(df['epoch'], df['total'].dropna(), label='total', linewidth=0.5, linestyle='-')
    else:
        ax.plot(df.index, df['total'].dropna(), label='total', linewidth=0.5, linestyle='-')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.legend()
    if log_scale:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "total_loss_plot.png", dpi=150)
        print(f"Saved to {save_dir}")

    #plt.show()


# TODO: MAGGIE - this would be the plot comparing predictions
#  (it can call other places if you want, just need the function to pass through here)
def plot_paired_predictions(preds_np, truth_np, xs, ys, title='', save_dir=None, save_suffix=''):
    # place holder --> could replace w something like
    # ground_truth_class.maggie_plot_function(loss, preds_np, xs, ys)
    # preds_np = preds_np / 200 + 25
    # Calculate absolute error
    error_np = np.abs(preds_np - truth_np)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Use a shared scale for Pred and Truth for an honest visual comparison
    vmin = min(preds_np.min(), truth_np.min())
    vmax = max(preds_np.max(), truth_np.max())
    levels = np.linspace(vmin, vmax, 50)

    # Panel 1: Prediction
    im0 = axes[0].contourf(xs, ys, preds_np, levels=levels, cmap='hot')
    axes[0].set_title(f'PINN Prediction\nRange: [{preds_np.min():.1f}, {preds_np.max():.1f}]')
    fig.colorbar(im0, ax=axes[0])

    # Panel 2: Analytical (Ground Truth)
    im1 = axes[1].contourf(xs, ys, truth_np, levels=levels, cmap='hot')
    axes[1].set_title(f'Analytical Truth\nRange: [{truth_np.min():.1f}, {truth_np.max():.1f}]')
    fig.colorbar(im1, ax=axes[1])

    # Panel 3: Absolute Error
    # Using 'viridis' or 'plasma' helps highlight where the delta is highest
    im2 = axes[2].contourf(xs, ys, error_np, levels=50, cmap='viridis')
    axes[2].set_title(f'Absolute Error\nMean: {np.mean(error_np):.2e}')
    fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_dir:
        path = save_dir / f'analytical_comparison{save_suffix}.png'
        plt.savefig(path, dpi=150)
        print(f'Saved comparison to {path}')

    #plt.show()


# TODO: MAGGIE CONTEXT -->  - below are plot methods mostly for trouble shooting
#  welcome to change em up, but no work needed here right now as far as I know

def plot_predicted_temperature(temps, xs, ys, title='', save_dir=None, save_suffix=''):
    """
    Plots predicted temperature field from the PINN model.
    """
    # temps = temps / 200 + 25
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.contourf(xs, ys, temps, levels=50, cmap='hot')
    # ax.scatter(power_source.x, power_source.y, c='cyan', marker='x', s=100, label='Heat source')
    # ax.set_title(f'Predicted Temperature ({title}) (Loss: {loss.item()})')
    ax.set_title(f'Predicted Temperature ({title})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.colorbar(im, ax=ax, label='Temp')
    plt.tight_layout()

    if save_dir:
        path = save_dir / f'predicted_temperature{save_suffix}.png'
        plt.savefig(path, dpi=150)
        print(f'Saved to {path}')

    #plt.show()


def plot_power_map_predictions(temps, power_np, xs, ys, title, save_dir=None, save_suffix=''):
    """
    Plots predicted vs analytical temperature fields side by side.
    Analytical solution is approximate: steady-state with Gaussian source + Robin BCs.
    """
    # temps = temps / 200 + 25
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].contourf(xs, ys, temps, levels=50, cmap='hot')
    axes[0].set_title(f'Predicted Temperature')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    fig.colorbar(im0, ax=axes[0], label='Temp')

    im1 = axes[1].contourf(xs, ys, power_np, levels=50, cmap='plasma')
    # axes[1].scatter(power_source.x, power_source.y, c='red', marker='x', s=100, label='Source center')
    axes[1].set_title(f'Power Map')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    # axes[1].legend()
    fig.colorbar(im1, ax=axes[1], label='Power (W)')

    plt.suptitle(title)
    plt.tight_layout()

    if save_dir:
        path = save_dir / f'temperature_comparison{save_suffix}.png'
        plt.savefig(path, dpi=150)
        print(f'Saved to {path}')

    #plt.show()

# TODO: MAGGIE - this one is just for guassian solutions,
#  its gross tbh but it lets me compare the temp range mostly
def plot_gauss_approx_solution(model, plate, grid, grid_map, power_source, save_dir=None, suffix=''):
    """
    Plots predicted vs analytical temperature fields side by side.
    Analytical solution is approximate: steady-state with Gaussian source + Robin BCs.
    """
    model.model.eval()

    bc = GaussianPde()
    bc.build_power_map(grid_map, [power_source], model._device)
    power_np = bc.power_map.cpu().detach().numpy().reshape(model.grid.shape())
    mod_in = model._build_input(bc.power_map)
    preds = model._model(mod_in)
    preds_np = preds.detach().cpu().numpy().reshape(model.grid.shape())

    xs = np.linspace(0, plate.length, grid.length)
    ys = np.linspace(0, plate.width, grid.width)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].contourf(xs, ys, preds_np, levels=50, cmap='hot')
    axes[0].set_title('Predicted Temperature')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    fig.colorbar(im0, ax=axes[0], label='Temp')

    im1 = axes[1].contourf(xs, ys, power_np, levels=50, cmap='plasma')
    axes[1].scatter(power_source.x, power_source.y, c='red', marker='x', s=100, label='Source center')
    axes[1].set_title(f'Power Input  (A={power_source.amplitude}, σ={power_source.spread})')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].legend()
    fig.colorbar(im1, ax=axes[1], label='Power (W)')

    # Approximate analytical temperature: convolve Gaussian source with 2D free-space
    # Green's function G(r) = -1/(2πk) * ln(r). Discretized as a sum over grid points.
    # This ignores boundary conditions — useful as a sanity check on the field shape.
    xx, yy = np.meshgrid(xs, ys)
    x0, y0 = power_source.x, power_source.y
    A, sigma = power_source.amplitude, power_source.spread
    k = plate.conduction

    # Q(x,y) at every grid point
    Q = A * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))

    # Convolve Q with Green's function: T(x,y) = sum_j Q(xj,yj) * G(|(x,y)-(xj,yj)|) * dA
    dx = plate.length / (grid.length - 1)
    dy = plate.width / (grid.width - 1)
    dA = dx * dy

    coords_x = xx.ravel()
    coords_y = yy.ravel()
    Q_flat = Q.ravel()

    T_flat = np.zeros_like(coords_x)
    eps = 1e-6  # avoid log(0) at source point
    rx = coords_x[:, None] - coords_x[None, :]  # (N, N)
    ry = coords_y[:, None] - coords_y[None, :]
    r = np.sqrt(rx ** 2 + ry ** 2)
    r = np.maximum(r, eps)
    G = -1.0 / (2 * np.pi * k) * np.log(r)  # (N, N)
    T_flat = (G * Q_flat[None, :] * dA).sum(axis=1)  # (N,)

    T_approx = T_flat.reshape(xx.shape)

    im2 = axes[2].contourf(xs, ys, T_approx, levels=50, cmap='hot')
    axes[2].scatter(power_source.x, power_source.y, c='red', marker='x', s=100, label='Source center')
    axes[2].set_title('Approx. Analytical Temp\n(free-space Green\'s fn, no BCs)')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].legend()
    fig.colorbar(im2, ax=axes[2], label='Temp')

    plt.suptitle(f'Gaussian source at ({power_source.x}, {power_source.y}), '
                 f'A={power_source.amplitude}, σ={power_source.spread}')
    plt.tight_layout()

    if save_dir:
        path = save_dir / f'temperature_comparison_{suffix}.png'
        plt.savefig(path, dpi=150)
        print(f'Saved to {path}')

    #plt.show()

def plot_pde_residuals(residuals_np, xs, ys, title='', save_dir=None, save_suffix=''):
    fig, ax = plt.subplots(figsize=(7, 6))
    max_bound = np.max(np.abs(residuals_np))

    # Avoid a divide-by-zero or flat scale if residuals are exactly 0 everywhere
    if max_bound == 0:
        max_bound = 1.0

    # 'RdBu_r' gives Red for positive, Blue for negative, White for 0
    im = ax.contourf(
        xs, ys, residuals_np,
        levels=50,
        cmap='RdBu_r',
        vmin=-max_bound,
        vmax=max_bound
    )

    ax.set_title(f'PDE Residual Map ({title})\nRange: [{np.min(residuals_np):.2e}, {np.max(residuals_np):.2e}]')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    cbar = fig.colorbar(im, ax=ax, label='k∇²u + Q')
    plt.tight_layout()

    if save_dir:
        path = save_dir / f'pde_residuals{save_suffix}.png'
        plt.savefig(path, dpi=150)
        print(f'Saved signed residual map to {path}')

    #plt.show()