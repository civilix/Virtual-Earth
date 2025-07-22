import numpy as np

m1 = np.array([[13, 5],
               [2,  4]])
m2 = np.array([[5,  2, 4],
               [-3, 6, 2],
               [3, -3, 1]])
m3 = np.array([[2, 1, 1],
               [2, 3, 2],
               [3, 3, 4]])

for idx, m in enumerate((m1, m2, m3), 1):
    w, v = np.linalg.eig(m)
    for i in range(len(w)):
        v[:, i] = v[:, i] / np.linalg.norm(v[:, i])
    print(f"Matrix {idx}")
    print("Eigenvalues:", w)
    print("Eigenvectors:\n", v)
    print("Trace:", np.trace(m), "Sum of eigenvalues:", np.sum(w))
