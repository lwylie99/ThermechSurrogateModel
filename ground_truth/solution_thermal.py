#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 06:51:53 2026

@author: maggiepoulsen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# =============================================================================
# 1. CONFIGURATION BLOCK (INPUT PARAMETERS)
# =============================================================================

# Domain Geometry
L = 40  # Length of the domain in the x-direction [mm]
W = 40  # Length of the domain in the y-direction [mm]

# Material Properties & Ambient Conditions
k = 0.3    # Thermal conductivity of FR-4 [W/(m·K)]
Ta = 25.0  # Ambient temperature [°C]

# Boundary Conditions (BCs)
# Note: For Neumann use val (thermal conductivity) = 0 (fully insulated) - non-zero heat flux not functional
#       For Robin use val (convection coefficient) = desired value (h) [W/(m²·K)]


BC_LEFT   = {"type": "Neumann",   "val": 0}
BC_RIGHT  = {"type": "Neumann",   "val": 0}
BC_BOTTOM = {"type": "Robin",       "val": 10.0}
BC_TOP    = {"type": "Robin",       "val": 10.0}

# Gaussian Heat Sources (Up to 4 sources)
# Input source values
power = 0.8     # Power [W] (~0.1 to 1.0) 
loc_x = 0.5     # Relative location in x-direction [%] (0 to 1) 
loc_y = 0.5     # Relative location in y-direction [%] (0 to 1)
sigma = 1.5    # Spread, effective width about x3 [mm] (1 to 2)

# Convert mm to m
L = L * 1e-3
W = W * 1e-3
sigma = sigma * 1e-3

# Format: [Amplitude (Aj), x_center, y_center, spread (sigma)]
sources = [
    [power / (2*np.pi * sigma**2),   loc_x * L,  loc_y * W,  sigma],  # Source 1
    [0.0,   0.0,      0.0,      1.0],   # Source 2: Unused (Amplitude = 0)
    [0.0,   0.0,      0.0,      1.0],   # Source 3: Unused (Amplitude = 0)
    [0.0,   0.0,      0.0,      1.0]    # Source 4: Unused (Amplitude = 0)
]

# Resolution Control (Number of terms in Fourier expansion)
N_max = 50  # x-direction terms
M_max = 50  # y-direction terms
Nx, Ny = 200, 200  # Mesh grid resolution for plotting

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
    return (beta_1 + beta_2) * w * np.cos(w * L) + (beta_1 * beta_2 - w**2) * np.sin(w * L)

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
        w_end = float(w_scan[idx+1])
        
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
    return (L / 2.0) * (1.0 + c**2) + (np.sin(2.0 * w * L) / (4.0 * w)) * (1.0 - c**2) + (c / w) * (np.sin(w * L)**2)

# Calculate eigenvalue arrays
lambdas = get_eigenvalues(beta_L, beta_R, L, N_max)
mus     = get_eigenvalues(beta_B, beta_T, W, M_max)

# =============================================================================
# 4. FIELD RECONSTRUCTION (SERIES SUMMATION)
# =============================================================================

# Setup spatial grid matrices
x_vec = np.linspace(0, L, Nx)
y_vec = np.linspace(0, W, Ny)
X, Y = np.meshgrid(x_vec, y_vec)

# Initialize array with ambient background temperature
T = np.zeros_like(X) + Ta

print("Computing fully flexible 2D eigenexpansion grid...")
for n, lam_n in enumerate(lambdas):
    # Skip trivial combinations if they don't apply
    if lam_n == 0.0 and beta_L != 0.0: 
        continue
    N_n = eval_norm(lam_n, beta_L, L)
    X_n = eval_eigenfunction(X, lam_n, beta_L)
    
    for m, mu_m in enumerate(mus):
        if mu_m == 0.0 and beta_B != 0.0: 
            continue
        if lam_n == 0.0 and mu_m == 0.0: 
            continue # Safe-guard division by zero
            
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
            source_contribution = (A_j * (sigma_j**2) * np.exp(-0.5 * (lam_n**2 + mu_m**2) * sigma_j**2) * X_n_xj * Y_m_yj)
            source_sum += source_contribution
            
        # Complete Fourier-Coefficient a_nm computation
        eig_sum = lam_n**2 + mu_m**2

        if eig_sum < 1e-12:
            continue

        denominator = k * eig_sum * N_n * M_m
        a_nm = (2.0 * np.pi / denominator) * source_sum
        
        # Project spatial Y function across the grid
        Y_m_Y = eval_eigenfunction(Y, mu_m, beta_B)
        
        # Accumulate component superposition into total matrix
        T += a_nm * X_n * Y_m_Y

print("Done! Generating visualization...")

# =============================================================================
# 5. POST-PROCESSING & VISUALIZATION
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 6.5))

contour = ax.contourf(X, Y, T, levels=65, cmap='turbo')
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label('Temperature [°C]', fontsize=11)

ax.set_title('Symmetric 2D Generalized Robin Analytical Solution',
             fontsize=13, fontweight='bold')
ax.set_xlabel('X Dimension [mm]', fontsize=11)
ax.set_ylabel('Y Dimension [mm]', fontsize=11)

bc_summary = (
    f"Boundary States:\n"
    f" Left: {BC_LEFT['type']} (h={h_L}, q={q_L})  |   "
    f"Right: {BC_RIGHT['type']} (h={h_R}, q={q_R})  |  "
    f"Bottom: {BC_BOTTOM['type']} (h={h_B}, q={q_B})  |  "
    f"Top: {BC_TOP['type']} (h={h_T}, q={q_T}) "
)

fig.text(
    0.5, 0.08,
    bc_summary,
    ha='center',
    va='bottom',
    fontsize=9,
    bbox=dict(boxstyle="round",
              facecolor="white",
              edgecolor="gray",
              alpha=0.8)
)

plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.show()