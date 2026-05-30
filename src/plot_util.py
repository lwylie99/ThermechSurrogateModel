import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from components_thermal import GaussianPde
from loss import PartLoss


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

def plot_predicted_temperature(model, grid, plate, power_source, save_dir=None):
    """
    Plots predicted temperature field from the PINN model.
    """
    model.model.eval()

    with torch.no_grad():
        grid_map = grid.build_grid_map(plate).reshape(-1, 2).to(model.device)
        preds = model.model_plate(power_source, coords=grid_map)
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


def plot_temperature_comparison(model, grid, plate, power_source, ambient=100, save_dir=None):
    """
    Plots predicted vs analytical temperature fields side by side.
    Analytical solution is approximate: steady-state with Gaussian source + Robin BCs.
    """
    model.model.eval()

    with torch.no_grad():
        grid_map = grid.build_grid_map(plate).reshape(-1, 2).to(model.device)
        preds = model.model_plate(power_source, coords=grid_map)
        preds_np = preds.detach().cpu().numpy().reshape(grid.length, grid.width)

    # approximate analytical: just the gaussian source integrated (not true steady state)
    xs = np.linspace(0, plate.length, grid.length)
    ys = np.linspace(0, plate.width, grid.width)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')

    r2 = (xx - power_source.x) ** 2 + (yy - power_source.y) ** 2
    Q = power_source.amplitude * np.exp(-r2 / (2 * power_source.spread ** 2))

    # rough steady state: ambient + scaled Q (not a true PDE solve)
    T_approx = ambient + Q / power_source.amplitude

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].contourf(xs, ys, preds_np, levels=50, cmap='hot')
    axes[0].set_title('Predicted Temperature')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0], label='Temp')

    im1 = axes[1].contourf(xs, ys, T_approx.T, levels=50, cmap='hot')
    axes[1].set_title('Approximate Analytical (All Sides Ambient)')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1], label='Temp')

    plt.suptitle(f'Gaussian source at ({power_source.x}, {power_source.y}), '
                 f'A={power_source.amplitude}, σ={power_source.spread}')
    plt.tight_layout()

    if save_dir:
        path = save_dir / 'temperature_comparison.png'
        plt.savefig(path, dpi=150)
        print(f'Saved to {path}')
    plt.show()


def plot_power_map(model, grid, plate, power_source, save_dir=None):
    """
    Plots the Gaussian power source distribution over the plate.
    """
    grid_map = grid.build_grid_map(plate).reshape(-1, 2).to(model.device)

    bc = GaussianPde()
    bc.build_power_map(grid_map, [power_source], model.device)
    power_np = bc.power_map.cpu().detach().numpy().reshape(grid.length, grid.width)

    xs = np.linspace(0, plate.length, grid.length)
    ys = np.linspace(0, plate.width, grid.width)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.contourf(xs, ys, power_np, levels=50, cmap='plasma')
    ax.scatter(power_source.x, power_source.y, c='white', marker='x', s=100, label='Source center')
    ax.set_title(f'Power Input  (A={power_source.amplitude}, σ={power_source.spread})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.colorbar(im, ax=ax, label='Power (W)')
    plt.tight_layout()

    if save_dir:
        path = save_dir / 'power_map.png'
        plt.savefig(path, dpi=150)
        print(f'Saved to {path}')
    plt.show()
