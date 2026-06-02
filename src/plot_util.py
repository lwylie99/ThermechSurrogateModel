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


def plot_predictions(model, plate, grid, grid_map, power_source, ambient=100, save_dir=None):
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].contourf(xs, ys, preds_np, levels=50, cmap='hot')
    axes[0].set_title('Predicted Temperature')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    # axes[0].colorbar(im0, ax=axes[0], label='Temp')

    bc = GaussianPde()
    bc.build_power_map(grid_map, [power_source], model.device)
    power_np = bc.power_map.cpu().detach().numpy().reshape(grid.length, grid.width)

    im1 = axes[1].contourf(xs, ys, power_np, levels=50, cmap='plasma')
    axes[1].scatter(power_source.x, power_source.y, c='white', marker='x', s=100, label='Source center')
    # axes[1].colorbar(im1, ax=axes[1], label='Power (W)')
    axes[1].set_title(f'Power Input  (A={power_source.amplitude}, σ={power_source.spread})')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].legend()

    plt.suptitle(f'Gaussian source at ({power_source.x}, {power_source.y}), '
                 f'A={power_source.amplitude}, σ={power_source.spread}')
    plt.tight_layout()

    if save_dir:
        path = save_dir / 'temperature_comparison.png'
        plt.savefig(path, dpi=150)
        print(f'Saved to {path}')

        path = save_dir / 'power_map.png'
        plt.savefig(path, dpi=150)
        print(f'Saved to {path}')

    plt.show()
