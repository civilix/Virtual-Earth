import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Domain parameters
Lx, Ly = 1.0, 1.0
Nx, Ny = 50, 50
dx, dy = Lx/(Nx-1), Ly/(Ny-1)

# Diffusion coefficient
D = 0.01

# 1D diffusion matrix with Neumann BC (no-flux)
def diffusion_1d_neumann(N, d):
    main = -2.0 * np.ones(N)
    main[0] = -1.0  # Neumann at boundaries
    main[-1] = -1.0
    off = np.ones(N-1)
    return sp.diags([off, main, off], offsets=[-1, 0, 1], format='csr') / d**2

# Build 2D Laplacian
Ix = sp.eye(Nx, format='csr')
Iy = sp.eye(Ny, format='csr')
Ax = diffusion_1d_neumann(Nx, dx)
Ay = diffusion_1d_neumann(Ny, dy)
Laplacian = sp.kron(Iy, Ax, format='csr') + sp.kron(Ay, Ix, format='csr')

# Initial condition: Gaussian blob at center
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
sigma = 0.05
c = np.exp(-((X-0.5)**2 + (Y-0.5)**2) / (2 * sigma**2)).flatten()

# Time-stepping parameters
dt = dx * dy / (4 * D) * 0.4  # stability
n_steps = 100

# Set up figure
fig, ax = plt.subplots()
img = ax.imshow(c.reshape((Ny, Nx)), origin='lower', extent=(0, Lx, 0, Ly))
ax.set_title('Pure Diffusion t = 0.00')
ax.set_xlabel('x')
ax.set_ylabel('y')
cbar = fig.colorbar(img, ax=ax, label='Concentration')

def update(frame):
    global c
    c = c + D * dt * Laplacian.dot(c)
    C = c.reshape((Ny, Nx))
    img.set_data(C)
    ax.set_title(f'Pure Diffusion t = {frame*dt:.2f}')
    return img,

ani = FuncAnimation(fig, update, frames=n_steps, interval=100, blit=True)
ani.save("diffusion.gif", writer="Pillow")