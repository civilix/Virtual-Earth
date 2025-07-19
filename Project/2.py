# -*- coding: utf-8 -*-
"""
Simulation of Flatus Dispersion with Initial Velocity in a 2D Confined Space.

VERSION 3: Stable Upwind Scheme

This version replaces the unstable Central Difference Scheme with a First-Order
Upwind Scheme for the advection term. This eliminates numerical oscillations
and provides a much more physically realistic simulation of the gas cloud.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Simulation Parameters (Identical to before) ---
WIDTH, HEIGHT = 10.0, 10.0
NX, NY = 101, 101
TOTAL_TIME = 200.0
DT = 0.1 # Using a slightly smaller DT can also improve stability
NT = int(TOTAL_TIME / DT)
D = 0.01
C0 = 1.0
X0, Y0 = 2.0, 5.0
SIGMA = 0.2
UX_INITIAL = 0.5
UY_INITIAL = 0.0
DECAY_RATE = 0.01

# --- 2. Setup the Numerical Grid (Identical to before) ---
dx = WIDTH / (NX - 1)
dy = HEIGHT / (NY - 1)
x = np.linspace(0, WIDTH, NX)
y = np.linspace(0, HEIGHT, NY)
X, Y = np.meshgrid(x, y)

# --- 3. Set Initial Conditions (Identical to before) ---
C = np.zeros((NY, NX))
C = C0 * np.exp(-((X - X0)**2 + (Y - Y0)**2) / (2 * SIGMA**2))

ux = np.full((NY, NX), UX_INITIAL)
uy = np.full((NY, NX), UY_INITIAL)
ux[0, :] = 0; ux[-1, :] = 0; ux[:, 0] = 0; ux[:, -1] = 0
uy[0, :] = 0; uy[-1, :] = 0; uy[:, 0] = 0; uy[:, -1] = 0

def simulate_step_stable(C, ux, uy):
    """
    Performs one time step using a STABLE UPWIND SCHEME for advection.
    """
    Cn = C.copy()

    # --- ★★★ KEY CHANGE: UPWIND SCHEME for Advection Term ★★★ ---
    # The diffusion term remains a central difference scheme.
    # We assume ux is always positive and uy is zero in this specific problem.
    # A general implementation would check the sign of u at each point.

    advection_x = ux[1:-1, 1:-1] * DT / dx * (Cn[1:-1, 1:-1] - Cn[1:-1, 0:-2])

    # Since uy is zero, advection_y is zero, but we write it for completeness.
    # This assumes uy is positive. A full implementation needs a check.
    advection_y = uy[1:-1, 1:-1] * DT / dy * (Cn[1:-1, 1:-1] - Cn[0:-2, 1:-1])

    diffusion_x = D * DT / dx**2 * (Cn[1:-1, 2:] - 2 * Cn[1:-1, 1:-1] + Cn[1:-1, 0:-2])
    diffusion_y = D * DT / dy**2 * (Cn[2:, 1:-1] - 2 * Cn[1:-1, 1:-1] + Cn[0:-2, 1:-1])

    C[1:-1, 1:-1] = Cn[1:-1, 1:-1] - advection_x - advection_y + diffusion_x + diffusion_y
    # --- End of Key Change ---

    # Apply No-Flux Neumann Boundary Conditions for Concentration
    C[0, :] = C[1, :]
    C[-1, :] = C[-2, :]
    C[:, 0] = C[:, 1]
    C[:, -1] = C[:, -2]
    return C

# --- Animation Setup ---
fig, ax = plt.subplots(figsize=(8, 7))
pcm = ax.pcolormesh(X, Y, C, cmap='viridis', vmin=0, vmax=C0)
fig.colorbar(pcm, ax=ax, label='Concentration')
ax.set_title("Gas Dispersion at t = 0.0 s")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_aspect('equal', adjustable='box')

def update(frame):
    global C, ux, uy
    current_time = frame * DT

    current_ux_val = UX_INITIAL * np.exp(-DECAY_RATE * current_time)
    current_uy_val = UY_INITIAL * np.exp(-DECAY_RATE * current_time)

    ux[1:-1, 1:-1] = current_ux_val
    uy[1:-1, 1:-1] = current_uy_val

    # Use the new stable simulation function
    C = simulate_step_stable(C, ux, uy)

    pcm.set_array(C.ravel())
    ax.set_title(f"Stable Simulation at t = {current_time:.1f} s")

    return [pcm]

print("Starting stable simulation with Upwind Scheme...")
anim = FuncAnimation(fig, update, frames=NT, interval=30, blit=True)
anim.save('fart_dispersion_stable.gif', writer='pillow', fps=25)
plt.close()
print("Stable animation saved as 'fart_dispersion_stable.gif'")