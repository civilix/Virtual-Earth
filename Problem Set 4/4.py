import numpy as np
import matplotlib.pyplot as plt

# 4a
ts = 300
dx = 1
x = np.zeros(ts + 1)
t = np.arange(ts + 1)
for i in range(1, ts + 1):
    r = np.random.rand()
    if r < 0.5:
        x[i] = x[i-1] - dx
    else:
        x[i] = x[i-1] + dx
plt.plot(t, x)
plt.show()

# 4b
np = 1000
ts = 300
dx = 1
t = np.arange(ts + 1)
pos = np.zeros((np, ts + 1))
msq_pos = np.zeros(ts + 1)

for i in range(1, ts + 1):
    r = np.random.rand(np)
    steps = np.where(r < 0.5, -dx, dx)
    pos[:, i] = pos[:, i-1] + steps
    msq_pos[i] = np.mean(pos[:, i]**2)

plt.plot(t, pos.T)
plt.show()

plt.plot(t, msq_pos)
plt.show()

# 4c
final_pos = pos[:, -1]
plt.hist(final_pos, bins=50)
plt.show()