import numpy as np
from utils import *
def exact(tensor):
    N = tensor.shape[2]
    B = decimal2binary(list(range(2 ** N)), N)
    vopt = - np.inf
    for i in range(B.shape[1]):
        b = B[:, i, None]
        Y = yofb(tensor, b)
        v = lambda_max(Y)
        if v > vopt: vopt, bopt = v, b 
    qopt = phi(yofb(tensor, bopt))
    return qopt, bopt, vopt

def yofb(tensor, b):
    D = tensor.shape[0]
    N = b.shape[0]
    Y = np.zeros((D, D))
    for n in range(N):
        Y = Y + b[n] * 0.5 * (tensor[:, :, n] + tensor[:, :, n].T)
    return Y