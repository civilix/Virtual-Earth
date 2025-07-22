import numpy as np
m=np.array([[0,1,0],[1,0,0],[0,0,1]])
w,v=np.linalg.eigh(m)
vals,count=np.unique(w,return_counts=True)
gm=[3-np.linalg.matrix_rank(m - val*np.eye(3)) for val in vals]
print("eigenvalues:",w)
print("eigenvectors:\n",v)
print("algebraic multiplicities:",dict(zip(vals,count)))
print("geometric multiplicities:",dict(zip(vals,gm)))
print("Q^T Q:\n",v.T.dot(v))
print("QΛQ^T:\n",v.dot(np.diag(w)).dot(v.T))
