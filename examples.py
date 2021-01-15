import algorithms as arescal
import numpy as np
import matplotlib.pyplot as plt
import imagesc as imagesc
D = 10
N = 10
tensor = np.random.randn(D, D, N)
# Compute exact solution by exhaustive search
qopt, bopt, vopt = arescal.exact(tensor)
print('Metric attained by exact solution:\t\t' + str(vopt))
# Compute approximate solution by fixed-point iterations
q, b, evo = arescal.efficient(tensor) 
print('Metric attained by approximate solution:\t' + str(evo[-1]))
# Illustrate metric evolution
plt.plot(range(len(evo)), evo, label = 'Approximate solution')
plt.plot([1, len(evo)], [vopt, vopt], label = 'Exact solution')
plt.ylabel('Metric')
plt.xlabel('Iteration index')
plt.legend()
plt.show()
# Approximate multiple component using fixed-point iterations+deflation
Q = arescal.efficient_deflation(tensor, 3) 
fig = imagesc.seaborn(Q.T @ Q)