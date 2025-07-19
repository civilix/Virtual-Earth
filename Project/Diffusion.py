import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class FlatusSimulator:
    SPECTATOR_LOCATIONS = [
        (0.5, 1.5), (1.0, 1.5), (1.5, 1.5),
        (0.5, 1.0), (1.0, 1.0), (1.5, 1.0),
        (0.5, 0.5), (1.0, 0.5), (1.5, 0.5),
    ]
    COLOR_SAFE = 'grey'
    COLOR_DETECTED = 'red'

    def __init__(self, source_choice_1_to_9: int, wind_angle_degrees: float):
        internal_source_index = source_choice_1_to_9 - 1
        self.WIDTH, self.HEIGHT = 2.0, 2.0
        self.NX, self.NY = 201, 201
        self.DT = 0.02
        self.TOTAL_TIME = 10.0
        self.NT = int(self.TOTAL_TIME / self.DT)
        self.D = 0.01
        self.C0 = 1.0
        self.SIGMA = 0.075
        self.INITIAL_SPEED = 1.0
        self.WIND_DURATION = 2.0
        self.CONCENTRATION_THRESHOLD = 0.03
        angle_rad = np.deg2rad(wind_angle_degrees)
        self.UX_INITIAL = self.INITIAL_SPEED * np.cos(angle_rad)
        self.UY_INITIAL = self.INITIAL_SPEED * np.sin(angle_rad)
        all_coords = list(self.SPECTATOR_LOCATIONS)
        self.X0, self.Y0 = all_coords.pop(internal_source_index)
        self.detector_coords = np.array(all_coords)
        self.dx = self.WIDTH / (self.NX - 1)
        self.dy = self.HEIGHT / (self.NY - 1)
        self.X, self.Y = np.meshgrid(np.linspace(0, self.WIDTH, self.NX), np.linspace(0, self.HEIGHT, self.NY))
        self.C = self.C0 * np.exp(-((self.X - self.X0)**2 + (self.Y - self.Y0)**2) / (2 * self.SIGMA**2))
        self.ux = np.full((self.NY, self.NX), self.UX_INITIAL)
        self.uy = np.full((self.NY, self.NX), self.UY_INITIAL)
        for v in [self.ux, self.uy]: v[0, :] = 0; v[-1, :] = 0; v[:, 0] = 0; v[:, -1] = 0
        self.detector_indices = np.array([(int(sp_y / self.dy), int(sp_x / self.dx)) for sp_x, sp_y in self.detector_coords])
        self.detector_triggered_state = np.full(len(self.detector_coords), False, dtype=bool)
        self.fig, self.ax, self.pcm, self.spectator_plot = None, None, None, None

    def _solve_diffusion_cn(self):
        C_n = self.C.copy()
        C_star = np.zeros_like(C_n)
        alpha_x = self.D * self.DT / (2 * self.dx**2)
        alpha_y = self.D * self.DT / (2 * self.dy**2)
        A = np.diag(-alpha_x * np.ones(self.NX - 3), -1) + \
            np.diag((1 + 2 * alpha_x) * np.ones(self.NX - 2), 0) + \
            np.diag(-alpha_x * np.ones(self.NX - 3), 1)
        for j in range(1, self.NY - 1):
            RHS = C_n[j, 1:-1] + alpha_y * (C_n[j+1, 1:-1] - 2*C_n[j, 1:-1] + C_n[j-1, 1:-1])
            C_star[j, 1:-1] = np.linalg.solve(A, RHS)
        B = np.diag(-alpha_y * np.ones(self.NY - 3), -1) + \
            np.diag((1 + 2 * alpha_y) * np.ones(self.NY - 2), 0) + \
            np.diag(-alpha_y * np.ones(self.NY - 3), 1)
        for i in range(1, self.NX - 1):
            RHS = C_star[1:-1, i] + alpha_x * (C_star[1:-1, i+1] - 2*C_star[1:-1, i] + C_star[1:-1, i-1])
            self.C[1:-1, i] = np.linalg.solve(B, RHS)

    def _simulate_step(self):
        Cn = self.C.copy()
        ux_int = self.ux[1:-1, 1:-1]; uy_int = self.uy[1:-1, 1:-1]
        diff_x_back = (Cn[1:-1, 1:-1] - Cn[1:-1, 0:-2]); diff_x_fwd  = (Cn[1:-1, 2:] - Cn[1:-1, 1:-1])
        diff_y_back = (Cn[1:-1, 1:-1] - Cn[0:-2, 1:-1]); diff_y_fwd  = (Cn[2:, 1:-1] - Cn[1:-1, 1:-1])
        advection_x = np.where(ux_int > 0, ux_int * diff_x_back, ux_int * diff_x_fwd)
        advection_y = np.where(uy_int > 0, uy_int * diff_y_back, uy_int * diff_y_fwd)
        self.C[1:-1, 1:-1] -= (self.DT / self.dx) * advection_x + (self.DT / self.dy) * advection_y
        self._solve_diffusion_cn()
        self.C[0, :] = self.C[1, :]; self.C[-1, :] = self.C[-2, :]
        self.C[:, 0] = self.C[:, 1]; self.C[:, -1] = self.C[:, -2]

    def _update_frame(self, frame):
        current_time = frame * self.DT
        if current_time < self.WIND_DURATION:
            decay_factor = 1.0 - (current_time / self.WIND_DURATION)
            current_ux_val = self.UX_INITIAL * decay_factor
            current_uy_val = self.UY_INITIAL * decay_factor
        else:
            current_ux_val = 0.0
            current_uy_val = 0.0
        self.ux[1:-1, 1:-1] = current_ux_val
        self.uy[1:-1, 1:-1] = current_uy_val
        self._simulate_step()
        self.pcm.set_array(self.C.ravel())
        self.ax.set_title(f"Source: {self.X0, self.Y0} | t = {current_time:.3f} s")
        for i, (iy, ix) in enumerate(self.detector_indices):
            if self.C[iy, ix] > self.CONCENTRATION_THRESHOLD:
                self.detector_triggered_state[i] = True
        new_colors = [self.COLOR_DETECTED if triggered else self.COLOR_SAFE for triggered in self.detector_triggered_state]
        self.spectator_plot.set_facecolors(new_colors)
        return [self.pcm, self.spectator_plot]

    def run(self, save_filename: str, playback_slowdown_factor: float = 1.0):
        target_fps = (1 / self.DT) / playback_slowdown_factor
        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self.pcm = self.ax.pcolormesh(self.X, self.Y, self.C, cmap='viridis', vmin=0, vmax=self.C0)
        self.fig.colorbar(self.pcm, ax=self.ax, label='Concentration')
        self.spectator_plot = self.ax.scatter(self.detector_coords[:, 0], self.detector_coords[:, 1], c=self.COLOR_SAFE, s=50, edgecolors='white', zorder=3)
        self.ax.set_title("Gas Dispersion at t = 0.0 s")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_aspect('equal', adjustable='box')
        anim = FuncAnimation(self.fig, self._update_frame, frames=self.NT, interval=20, blit=True)
        anim.save(save_filename, writer='pillow', fps=target_fps)
        plt.close(self.fig)

if __name__ == "__main__":
    SOURCE_CHOICE_1_to_9 = 1
    WIND_ANGLE_DEGREES = 300
    PLAYBACK_FACTOR = 1
    simulator = FlatusSimulator(
        source_choice_1_to_9=SOURCE_CHOICE_1_to_9,
        wind_angle_degrees=WIND_ANGLE_DEGREES
    )
    output_filename = f'src_{SOURCE_CHOICE_1_to_9}_ang_{int(WIND_ANGLE_DEGREES)}.gif'
    simulator.run(
        save_filename=output_filename,
        playback_slowdown_factor=PLAYBACK_FACTOR
    )