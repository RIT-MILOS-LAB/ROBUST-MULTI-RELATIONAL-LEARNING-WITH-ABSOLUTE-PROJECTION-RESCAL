import numpy as np
def lambda_max(matrix):
    return np.sort(np.linalg.eig(matrix)[0])[-1]