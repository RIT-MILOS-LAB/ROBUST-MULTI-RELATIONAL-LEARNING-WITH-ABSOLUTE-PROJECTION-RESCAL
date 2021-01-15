import numpy as np
from .efficient import efficient
def efficient_deflation(tensor, components):
    D = tensor.shape[0]
    Q = np.zeros((D, components))
    for i in range(components):
        tensori = deflate(tensor, Q)
        q = efficient(tensori)[0]
        Q[:, i] = q
    return Q

def deflate(tensor, Q):
    N = tensor.shape[2]
    D = tensor.shape[0]
    tensor_def = np.zeros((D, D, N))
    P = np.eye(D) - Q @ Q.T
    for n in range(N):
        tensor_def[:, :, n] = P @ tensor[:, :, n] @ P
    return tensor_def