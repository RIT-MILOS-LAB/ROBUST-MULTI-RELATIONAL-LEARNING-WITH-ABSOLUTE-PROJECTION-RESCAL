import numpy as np
def phi(matrix):
    vals, eigvec = np.linalg.eig(matrix)
    idx = np.argmax(vals)
    return eigvec[:, idx]