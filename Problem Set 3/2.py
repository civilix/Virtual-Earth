import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu
from 1 import DiffusionMatrix

# ------------- 参数定义 -------------
L = 1         # 棒长 [m]
kappa = 0.001   # 扩散系数 [m^2/s]
dx = 0.05       # 空间步长 [m]
T_left = 0    # 左端温度 [°C]
T_right = 10  # 右端温度 [°C]
t_end = 200     # 模拟结束时间 [s]

# 网格点
n_nodes = L//dx + 1         # 包含两端点的总节点数
x_full = np.linspace(0, L, n_nodes)
N = n_nodes - 2                 # 内部未知节点数
x = x_full[1:-1]                # 内部节点位置



M = kappa / dx**2 * DiffusionMatrix(N)

# 边界条件源项
b = np.zeros(N)
b[-1] = kappa / dx **2 * T_right

# 在指定 dt 和方法下模拟并绘图
def simulate_forward(dt):
    r = kappa * dt / dx**2
    times = np.arange(0, t_end+1, 20)
    T = np.zeros(N)
    T[x > L/2] = T_right
    sol = {0: T.copy()}
    steps = int(np.ceil(t_end / dt))
    for n in range(1, steps+1):
        T = T + dt * (M.dot(T) + b)
        t = n * dt
        # 记录每隔 20s 的解
        if np.isclose(t % 20, 0, atol=dt/2):
            sol[int(round(t))] = T.copy()
    # 绘制
    plt.figure()
    for tt in times:
        T_full = np.concatenate(([T_left], sol[tt], [T_right]))
        plt.plot(x_full, T_full, label=f"t={tt}s")
    plt.xlabel("x [m]")
    plt.ylabel("T [°C]")
    plt.title(f"Forward Euler (dt={dt}s, r={r:.2f})")
    plt.legend()

def simulate_backward(dt):
    r = kappa * dt / dx**2
    A_imp = eye(N, format='csc') - dt * M
    LU = splu(A_imp)
    times = np.arange(0, t_end+1, 20)
    T = np.zeros(N)
    T[x > L/2] = T_right
    sol = {0: T.copy()}
    steps = int(np.ceil(t_end / dt))
    for n in range(1, steps+1):
        rhs = T + dt * b
        T = LU.solve(rhs)
        t = n * dt
        if np.isclose(t % 20, 0, atol=dt/2):
            sol[int(round(t))] = T.copy()
    # 绘制
    plt.figure()
    for tt in times:
        T_full = np.concatenate(([T_left], sol[tt], [T_right]))
        plt.plot(x_full, T_full, label=f"t={tt}s")
    plt.xlabel("x [m]")
    plt.ylabel("T [°C]")
    plt.title(f"Backward Euler (dt={dt}s, r={r:.2f})")
    plt.legend()

# 前向欧拉法：稳定 (r<0.5)
simulate_forward(dt=1.0)

# 前向欧拉法：不稳定 (r>0.5)
simulate_forward(dt=2.0)

# 后向欧拉法：在不稳定区间 dt 情况下仍然稳定
simulate_backward(dt=2.0)

# 展示所有图像
plt.show()