import numpy as np
import scipy.sparse as sp

def DiffusionMatrix(n):
    """
    Constructs the diffusion matrix of size n x n with -2 on the diagonal
    and 1 on the sub- and super-diagonals (sparse format).
    """
    diagonals = [
        -2 * np.ones(n),      # Main diagonal
        np.ones(n - 1),       # Sub-diagonal
        np.ones(n - 1)        # Super-diagonal
    ]
    offsets = [0, -1, 1]      # Diagonal offsets: main (0), sub (-1), super (+1)
    A = sp.diags(diagonals, offsets, shape=(n, n), format='csr')
    return A

# Example usage:
n = 5
A = DiffusionMatrix(n)
print("Diffusion matrix (n=5):")
print(A.toarray())