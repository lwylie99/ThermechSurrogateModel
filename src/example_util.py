import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from components_thermal import GaussianPde


def eval_example(model, plate, grid, power_data: list, save_dir=None):
    for i in range(len(power_data)):
        plot_gauss_approx_solution(model, plate, grid, model.grid_map, power_data[i], save_dir=save_dir, suffix=f'_p{i}')

def train_example(model, plate, grid, power_data: list, epochs=10000, save_dir=None):
    print(f'\nBEGIN TRAINING ({epochs} Epochs)\n')
    loss_hist = model.train_model(power_data=power_data, epochs=epochs)
    print(loss_hist)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        loss_hist.to_csv(save_dir / "loss_history.csv")

    plot_bc_loss(loss_hist, log_scale=True, save_dir=save_dir)
    plot_total_loss(loss_hist, log_scale=True, save_dir=save_dir)

def plot_bc_loss(df: pd.DataFrame, log_scale=False, save_dir=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in df.columns:
        if col == 'total':
            continue
        ax.plot(df.index, df[col].dropna(), label=col, linewidth=1.5, linestyle='-')

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

    plt.show()


def plot_total_loss(df: pd.DataFrame, log_scale=False, save_dir=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['total'].dropna(), label='total', linewidth=2.5, linestyle='--')

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

    plt.show()


def plot_predicted_temperature(model, plate, grid, power_source, save_dir=None):
    """
    Plots predicted temperature field from the PINN model.
    """
    model.model.eval()

    with torch.no_grad():
        mod_in = model._build_input(power_source, coords=model.grid_map)
        preds = model._model(mod_in)
        preds_np = preds.detach().cpu().numpy().reshape(grid.length, grid.width)

    xs = np.linspace(0, plate.length, grid.length)
    ys = np.linspace(0, plate.width, grid.width)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.contourf(xs, ys, preds_np, levels=50, cmap='hot')
    ax.scatter(power_source.x, power_source.y, c='cyan', marker='x', s=100, label='Heat source')
    ax.set_title('Predicted Temperature')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.colorbar(im, ax=ax, label='Temp')
    plt.tight_layout()

    if save_dir:
        path = save_dir / 'predicted_temperature.png'
        plt.savefig(path, dpi=150)
        print(f'Saved to {path}')
    plt.show()


def plot_gauss_predictions(model, plate, grid, grid_map, power_source, save_dir=None, suffix=''):
    """
    Plots predicted vs analytical temperature fields side by side.
    Analytical solution is approximate: steady-state with Gaussian source + Robin BCs.
    """
    model.model.eval()
    with torch.no_grad():
        mod_in = model._build_input(power_source, coords=model.grid_map)
        preds = model._model(mod_in)
        preds_np = preds.detach().cpu().numpy().reshape(grid.length, grid.width)

    xs = np.linspace(0, plate.length, grid.length)
    ys = np.linspace(0, plate.width, grid.width)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    im0 = axes[0].contourf(xs, ys, preds_np, levels=50, cmap='hot')
    axes[0].set_title('Predicted Temperature')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    fig.colorbar(im0, ax=axes[0], label='Temp')

    bc = GaussianPde()
    bc.build_power_map(grid_map, [power_source], model.device)
    power_np = bc.power_map.cpu().detach().numpy().reshape(grid.length, grid.width)

    im1 = axes[1].contourf(xs, ys, power_np, levels=50, cmap='plasma')
    axes[1].scatter(power_source.x, power_source.y, c='red', marker='x', s=100, label='Source center')
    axes[1].set_title(f'Power Input  (A={power_source.amplitude}, σ={power_source.spread})')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].legend()
    fig.colorbar(im1, ax=axes[1], label='Power (W)')

    plt.suptitle(f'Gaussian source at ({power_source.x}, {power_source.y}), '
                 f'A={power_source.amplitude}, σ={power_source.spread}')
    plt.tight_layout()

    if save_dir:
        path = save_dir / f'temperature_comparison_{suffix}.png'
        plt.savefig(path, dpi=150)
        print(f'Saved to {path}')

    plt.show()

def plot_gauss_approx_solution(model, plate, grid, grid_map, power_source, save_dir=None, suffix=''):
    """
    Plots predicted vs analytical temperature fields side by side.
    Analytical solution is approximate: steady-state with Gaussian source + Robin BCs.
    """
    model.model.eval()
    with torch.no_grad():
        mod_in = model._build_input(power_source, coords=model.grid_map)
        preds = model._model(mod_in)
        preds_np = preds.detach().cpu().numpy().reshape(grid.length, grid.width)

    xs = np.linspace(0, plate.length, grid.length)
    ys = np.linspace(0, plate.width, grid.width)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].contourf(xs, ys, preds_np, levels=50, cmap='hot')
    axes[0].set_title('Predicted Temperature')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    fig.colorbar(im0, ax=axes[0], label='Temp')

    bc = GaussianPde()
    bc.build_power_map(grid_map, [power_source], model.device)
    power_np = bc.power_map.cpu().detach().numpy().reshape(grid.length, grid.width)

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

    plt.show()
