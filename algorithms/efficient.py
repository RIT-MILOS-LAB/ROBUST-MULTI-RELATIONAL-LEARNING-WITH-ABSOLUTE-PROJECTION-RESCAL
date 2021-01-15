import numpy as np
from utils import *
def efficient(tensor, initialization = None, tolerance = 1e-6):
    D = tensor.shape[0]
    N = tensor.shape[2]
    if initialization is None: initialization = omega(np.random.randn(D, ))
    q, u, = initialization, initialization
    b = update_b(tensor, q, u)
    evolution = [(q.T @ yofb(tensor, b) @ u).item()]
    while True:
        q = omega(yofb(tensor, b) @ u)
        b = update_b(tensor, q, u)
        u = omega(yofb(tensor, b) @ q)
        evolution.append((q.T @ yofb(tensor, b) @ u).item())
        if evolution[-1]-evolution[-2] < tolerance: break
    return q, b, evolution
 
def yofb(tensor, b):
    D = tensor.shape[0]
    N = b.shape[0]
    Y = np.zeros((D, D))
    for n in range(N):
        Y = Y + b[n] * 0.5 * (tensor[:, :, n] + tensor[:, :, n].T)
    return Y
def update_b(tensor, q, u):
    N = tensor.shape[2]
    b = np.empty((N, ))
    for n in range(N):
        b[n] = np.sign(q.T @ (tensor[:, :, n] + tensor[:, :, n].T) @ u)  
    return b