import numpy as np
import scipy.sparse as sp

def DiffusionMatrix(n):
    diagonals = [
        -2 * np.ones(n),
        np.ones(n),
        np.ones(n)
    ]
    offsets = [0, -1, 1]
    A = sp.diags(diagonals, offsets, shape=(n, n), format='csr')
    return A

print(DiffusionMatrix(5).toarray())
