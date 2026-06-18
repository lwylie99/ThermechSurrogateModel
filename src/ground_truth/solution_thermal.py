#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 06:51:53 2026

@author: maggiepoulsen
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

# =============================================================================
# 1. CONFIGURATION BLOCK (INPUT PARAMETERS)
# =============================================================================

# Domain Geometry
L = 40  # Length of the domain in the x-direction [mm]
W = 40  # Length of the domain in the y-direction [mm]

# Convert mm to m
L = L * 1e-3
W = W * 1e-3

# Material Properties & Ambient Conditions
k = 0.3  # Thermal conductivity of FR-4 [W/(m·K)]
Ta = 25.0  # Ambient temperature [°C]

# Resolution Control (Number of terms in Fourier expansion)
N_max = 50  # x-direction terms
M_max = 50  # y-direction terms
Nx, Ny = 48, 48  # Mesh grid resolution for input-output pairs
Nx_plot, Ny_plot = 200, 200

# Setup spatial grid matrices
x_vec = np.linspace(0, L, Nx)
y_vec = np.linspace(0, W, Ny)
X, Y = np.meshgrid(x_vec, y_vec)

x_plot = np.linspace(0, L, Nx_plot)
y_plot = np.linspace(0, W, Ny_plot)
X_plot, Y_plot = np.meshgrid(x_plot, y_plot)

# Boundary Conditions (BCs)
# Note: For Neumann use val (thermal conductivity) = 0 (fully insulated) - non-zero heat flux not functional
#       For Robin use val (convection coefficient) = desired value (h) [W/(m²·K)]


BC_LEFT = {"type": "Neumann", "val": 0.0}
BC_RIGHT = {"type": "Neumann", "val": 0.0}
BC_BOTTOM = {"type": "Robin", "val": 10.0}
BC_TOP = {"type": "Robin", "val": 10.0}

# Sampling plan for heat sources
sampling_plan = [
    {"num": "01", "label": "Center – Low Power & Med Spread", "power": 0.8, "x": 0.50, "y": 0.50, "sigma": 3.0},
    {"num": "02", "label": "Center – Med Power", "power": 1.0, "x": 0.50, "y": 0.50, "sigma": 1.0},
    {"num": "03", "label": "Center – High Power", "power": 1.2, "x": 0.50, "y": 0.50, "sigma": 1.0},
    {"num": "04", "label": "Center – Tight Spread", "power": 1.0, "x": 0.50, "y": 0.50, "sigma": 0.5},
    {"num": "05", "label": "Center – Broad Spread", "power": 1.0, "x": 0.50, "y": 0.50, "sigma": 3.0},
    {"num": "06", "label": "Bottom-Left Quarter", "power": 1.0, "x": 0.25, "y": 0.25, "sigma": 1.0},
    {"num": "07", "label": "Left Edge", "power": 1.0, "x": 0.10, "y": 0.50, "sigma": 1.0},
    {"num": "08", "label": "Top-Left Corner", "power": 1.0, "x": 0.10, "y": 0.90, "sigma": 1.0},
]

#sampling_plan = [sampling_plan[0]]


def case_filename(case):
    safe_label = (case["label"]
                  .replace(" – ", "_")
                  .replace("-", "")
                  .replace(" ", "_"))
    return f"Case{case['num']}_{safe_label}"


results = []
results_full = []
results_plot = []

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

input_dir = "inputs"
os.makedirs(input_dir, exist_ok=True)

# =============================================================================
# 2. VALIDATION & PARAMETER SETUP
# =============================================================================

# Extract physical parameters based on choices
h_L = BC_LEFT["val"] if BC_LEFT["type"] == "Robin" else 0.0
h_R = BC_RIGHT["val"] if BC_RIGHT["type"] == "Robin" else 0.0
h_B = BC_BOTTOM["val"] if BC_BOTTOM["type"] == "Robin" else 0.0
h_T = BC_TOP["val"] if BC_TOP["type"] == "Robin" else 0.0

q_L = BC_LEFT["val"] if BC_LEFT["type"] == "Neumann" else 0.0
q_R = BC_RIGHT["val"] if BC_RIGHT["type"] == "Neumann" else 0.0
q_B = BC_BOTTOM["val"] if BC_BOTTOM["type"] == "Neumann" else 0.0
q_T = BC_TOP["val"] if BC_TOP["type"] == "Neumann" else 0.0

# Verify steady-state physical validity
if h_L == 0.0 and h_R == 0.0 and h_B == 0.0 and h_T == 0.0:
    raise ValueError(
        "Physical Paradox Error: All 4 sides are set to Insulated. "
        "With active heat sources and no heat sink, steady-state temperature is infinite."
    )

# Compute Biot-like tracking parameters (beta = h/k)
beta_L, beta_R = h_L / k, h_R / k
beta_B, beta_T = h_B / k, h_T / k


# =============================================================================
# 3. MATHEMATICAL CORE FUNCTIONS
# =============================================================================


def characteristic_equation(w, beta_1, beta_2, L):
    """Symmetric transcendental equation for Robin-Robin boundaries."""
    return (beta_1 + beta_2) * w * np.cos(w * L) + (beta_1 * beta_2 - w ** 2) * np.sin(w * L)


def get_eigenvalues(beta_1, beta_2, L, num_terms):
    """Finds the required eigenvalues, cleanly handling pure insulated limits."""
    if beta_1 == 0.0 and beta_2 == 0.0:
        return np.array([n * np.pi / L for n in range(num_terms + 1)])

    roots = []
    max_w_estimate = (num_terms + 5) * np.pi / L

    # Grid scan to locate root crossings safely
    w_scan = np.linspace(1e-5, max_w_estimate, max(5000, num_terms * 500))
    f_scan = characteristic_equation(w_scan, beta_1, beta_2, L)

    # .tolist() converts the NumPy index array into native Python integers
    sign_changes = np.where(np.diff(np.sign(f_scan)))[0]

    for idx in sign_changes:
        # Crucial: cast to explicit python floats to ensure scalar behavior
        w_start = float(w_scan[idx])
        w_end = float(w_scan[idx + 1])

        f_start = characteristic_equation(w_start, beta_1, beta_2, L)
        f_end = characteristic_equation(w_end, beta_1, beta_2, L)

        # This will now safely evaluate because f_start and f_end are guaranteed scalars
        if np.sign(f_start) != np.sign(f_end):
            try:
                root = brentq(
                    characteristic_equation,
                    w_start,
                    w_end,
                    args=(float(beta_1), float(beta_2), float(L))
                )
                if root > 1e-5:
                    if len(roots) == 0 or not np.isclose(root, roots[-1], rtol=1e-4):
                        roots.append(root)
            except (ValueError, RuntimeError):
                continue

        if len(roots) == num_terms:
            break

    return np.array([0.0] + roots[:num_terms])


def eval_eigenfunction(coord, w, beta_1):
    """Evaluates the structural eigenfunction safely."""
    if np.isclose(w, 0.0):
        return np.ones_like(coord)
    if beta_1 == 0.0:
        return np.cos(w * coord)
    return np.cos(w * coord) + (beta_1 / w) * np.sin(w * coord)


def eval_norm(w, beta_1, L):
    """Calculates the normalization integral (inner product norm squared) analytically."""
    if np.isclose(w, 0.0):
        return L
    c = beta_1 / w
    return (L / 2.0) * (1.0 + c ** 2) + (np.sin(2.0 * w * L) / (4.0 * w)) * (1.0 - c ** 2) + (c / w) * (
            np.sin(w * L) ** 2)


def build_power_map(x, y, mus, sigmas, amplitudes, clip_threshold=1e-4):
    Xp, Yp = np.meshgrid(x, y, indexing="xy")
    P = np.zeros_like(Xp)

    for (x0, y0), sigma, A in zip(mus, sigmas, amplitudes):
        P += A * np.exp(
            -((Xp - x0) ** 2 + (Yp - y0) ** 2)
            / (2 * sigma ** 2)
        )

    P[P < clip_threshold] = 0.0
    return P


# =============================================================================
# 4. EXECUTE SAMPLE
# =============================================================================

# Gaussian Heat Sources
for case in sampling_plan:

    fname = case_filename(case)
    print(f"Running Case {case['num']} — {case['label']}")

    power = case["power"]
    loc_x = case["x"] * L
    loc_y = case["y"] * W
    sigma = case["sigma"] * 1e-3  # mm -> m

    # Format: [Amplitude (Aj), x_center, y_center, spread (sigma)]
    sources = [
        [power / (2 * np.pi * sigma ** 2), loc_x,
         loc_y, sigma],  # Source 1
        [0.0, 0.0, 0.0, 1.0],  # Source 2: Unused (Amplitude = 0)
        [0.0, 0.0, 0.0, 1.0],  # Source 3: Unused (Amplitude = 0)
        [0.0, 0.0, 0.0, 1.0]  # Source 4: Unused (Amplitude = 0)
    ]

    # Build powermap for ML input
    mus_sources = [
        (loc_x, loc_y)
    ]

    sigmas_sources = [
        sigma
    ]

    amps_sources = [
        power / (2 * np.pi * sigma ** 2)
    ]

    P = build_power_map(
        x_vec,
        y_vec,
        mus_sources,
        sigmas_sources,
        amps_sources
    )

    P_plot = build_power_map(
        x_plot,
        y_plot,
        mus_sources,
        sigmas_sources,
        amps_sources
    )

    # Calculate eigenvalue arrays
    lambdas = get_eigenvalues(beta_L, beta_R, L, N_max)
    mus = get_eigenvalues(beta_B, beta_T, W, M_max)

    # Initialize array with ambient background temperature
    T = np.zeros_like(X) + Ta
    T_plot = np.zeros_like(X_plot) + Ta

    # Field reconstruction (Series summation)

    for n, lam_n in enumerate(lambdas):
        # Skip trivial combinations if they don't apply
        if lam_n == 0.0 and beta_L != 0.0:
            continue
        N_n = eval_norm(lam_n, beta_L, L)
        X_n = eval_eigenfunction(X, lam_n, beta_L)
        X_n_plot = eval_eigenfunction(X_plot, lam_n, beta_L)

        for m, mu_m in enumerate(mus):
            if mu_m == 0.0 and beta_B != 0.0:
                continue
            if lam_n == 0.0 and mu_m == 0.0:
                continue  # Safe-guard division by zero

            M_m = eval_norm(mu_m, beta_B, W)

            # Superposition of narrow Gaussian source footprints
            source_sum = 0.0
            for A_j, x_j, y_j, sigma_j in sources:
                if A_j == 0.0:
                    continue

                # Evaluate baseline eigenfunctions at source center coordinates
                X_n_xj = eval_eigenfunction(x_j, lam_n, beta_L)
                Y_m_yj = eval_eigenfunction(y_j, mu_m, beta_B)

                # Narrow gaussian analytical scaling integration
                source_contribution = (
                        A_j * (sigma_j ** 2) * np.exp(-0.5 * (lam_n ** 2 + mu_m ** 2) * sigma_j ** 2) * X_n_xj * Y_m_yj)
                source_sum += source_contribution

            # Complete Fourier-Coefficient a_nm computation
            eig_sum = lam_n ** 2 + mu_m ** 2

            if eig_sum < 1e-12:
                continue

            denominator = k * eig_sum * N_n * M_m
            a_nm = (2.0 * np.pi / denominator) * source_sum

            # Project spatial Y function across the grid
            Y_m_Y = eval_eigenfunction(Y, mu_m, beta_B)
            Y_m_plot = eval_eigenfunction(Y_plot, mu_m, beta_B)

            # Accumulate component superposition into total matrix
            T += a_nm * X_n * Y_m_Y
            T_plot += a_nm * X_n_plot * Y_m_plot

    results.append({
        "Case": f"Case {case['num']}",
        "Label": case["label"],
        "Power_W": power,
        "x_mm": loc_x * 1e3,
        "y_mm": loc_y * 1e3,
        "Sigma": case["sigma"],
        "Tmax_C": np.max(T),
        "Tavg_C": np.mean(T),
        "Pmax": np.max(P),
    })

    results_full.append({
        "num": case["num"],
        "label": case["label"],
        "T": T.copy(),
        "power": power,
        "sigma": case["sigma"],
        "x_mm": loc_x * 1e3,
        "y_mm": loc_y * 1e3,
        "P": P.copy(),
    })

    results_plot.append({
        "T_plot": T_plot,
        "P_plot": P_plot,
        "num": case["num"],
        "label": case["label"],
    })

    # Individual case plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'box_aspect': 1})

    # Power Map (Input) - LEFT
    contour_P = ax1.contourf(
        X_plot * 1e3,
        Y_plot * 1e3,
        P_plot * 1e-6,
        levels=65,
        cmap="viridis"
    )

    cbar_P = fig.colorbar(
        contour_P,
        ax=ax1
    )
    cbar_P.set_label(
        "Power Density [W/mm²]",
        fontsize=11
    )

    ax1.set_title(
        "Input: Gaussian Power Map",
        fontsize=11,
        fontweight="bold"
    )

    ax1.set_xlabel("X [mm]")
    ax1.set_ylabel("Y [mm]")

    # Temperature Filed (Output) - RIGHT
    contour_T = ax2.contourf(X_plot * 1e3, Y_plot * 1e3, T_plot, levels=65, cmap="turbo")
    cbar = fig.colorbar(contour_T, ax=ax2)
    cbar.set_label("Temperature [°C]", fontsize=11)

    ax2.set_title(
        "Output: Steady-State Temperature",
        fontsize=11,
        fontweight="bold"
    )

    ax2.set_xlabel("X [mm]", fontsize=11)
    ax2.set_ylabel("Y [mm]", fontsize=11)

    # Overall Figure Details
    fig.suptitle(
        "FR-4 Substrate: 2D Thermal Problem\n"
        f"Case: {case['num']} | Source: {case['label']}"
        ,
        fontsize=13,
        fontweight="bold"
    )

    plot_summary = (
        r"$\bf{Boundary\ States}$"
        f":  Left: {BC_LEFT['type']} (h={h_L}, q={q_L})  |   "
        f"Right: {BC_RIGHT['type']} (h={h_R}, q={q_R})  |  "
        f"Bottom: {BC_BOTTOM['type']} (h={h_B}, q={q_B})  |  "
        f"Top: {BC_TOP['type']} (h={h_T}, q={q_T}) \n"
        r"$\bf{Source\ Details}$"
        f":  Power: {power:.1f} [W] |  Location: ({loc_x * 1e3:.1f}, {loc_y * 1e3:.1f})  |  Spread: {sigma * 1e3:.1f} [mm] "
    )

    fig.text(
        0.5, 0.06,
        plot_summary,
        ha='center',
        va='bottom',
        fontsize=9,
        bbox=dict(boxstyle="round",
                  facecolor="white",
                  edgecolor="gray",
                  alpha=0.8)
    )

    plt.tight_layout(rect=[0.0, 0.15, 1.0, 0.95])
    plt.savefig(os.path.join(output_dir, f"{fname}.png"), dpi=300, bbox_inches="tight")

    plt.close()

    print(f"  → Done  (T_max = {np.max(T):.2f} °C)")

# =============================================================================
# 5. SAVE RESULTS
# =============================================================================
df_output = pd.DataFrame(results)
df_output.to_csv(
    os.path.join(output_dir, "sampling_summary.csv"),
    index=False
)

all_T = np.array([item["T"] for item in results_full])
print(all_T.shape)
np.save("temperature.npy",
        all_T
        )

all_P = np.array([item["P"] for item in results_full]) * 1e-6
print(all_P.shape)
np.save("powermaps.npy",
        all_P
        )

# =============================================================================
# 6. SUMMARY FIGURE
# =============================================================================

fig, axes = plt.subplots(
    2, 4,
    figsize=(18, 9),
    subplot_kw={'box_aspect': 1},
    constrained_layout=True,
)
axes_flat = axes.flatten()

fig.suptitle(
    "Steady-State Temperature Fields — All Cases\nFR-4 Substrate (Analytical Solution)",
    fontsize=14,
    fontweight="bold",
    linespacing=1.6,
)

global_min = min(np.min(item["T_plot"]) for item in results_plot)
global_max = max(np.max(item["T_plot"]) for item in results_plot)

last_contour = None
for ax, item in zip(axes_flat, results_plot):
    last_contour = ax.contourf(
        X_plot * 1e3, Y_plot * 1e3, item["T_plot"],
        levels=65,
        cmap="turbo",
        vmin=global_min,
        vmax=global_max,
    )

    # Two-line title: bold case number on top, descriptive label below
    ax.set_title(
        f"Case {item['num']}\n{item['label']}",
        fontsize=8.5,
        fontweight="bold",
        linespacing=1.5,
    )
    ax.set_xlabel("X [mm]", fontsize=7.5)
    ax.set_ylabel("Y [mm]", fontsize=7.5)
    ax.tick_params(labelsize=7)

# Single colorbar anchored to the entire axes array — created ONCE, outside the loop
cbar = fig.colorbar(
    last_contour,
    ax=axes_flat.tolist(),  # shrinks all subplots uniformly
    orientation="vertical",
    fraction=0.015,
    pad=0.03,
)
cbar.set_label("Temperature [°C]", fontsize=11, labelpad=10)

plt.savefig(os.path.join(output_dir, "summary_all_cases.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\nAll cases complete. Results saved to:", output_dir)
