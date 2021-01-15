## ROBUST MULTI-RELATIONAL LEARNING WITH ABSOLUTE PROJECTION RESCAL


In this repo we implent algorithms for the exact and approximate solution to Absolute Projection RESCAL *(A-RESCAL)* as presented in [[1]](https://ieeexplore.ieee.org/document/8969097). Considering a collection of N matrix measurements ![eqation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cmathbf%20X_1%2C%20%5Cmathbf%20X_2%2C%20%5Cldots%2C%20%5Cmathbf%20X_N%2C%20%5Cmathbf%20X_n%20%5Cin%20%5Cmathbb%20R%5E%7BD%5Ctimes%20D%7D%20%5Cforall%20n), A-RESCAL is formally formulated as 

![equation](https://latex.codecogs.com/svg.latex?%5Cunderset%7B%5Cmathbf%20Q%20%5Cin%20%5Cmathbb%20R%5E%7BD%5Ctimes%20d%7D%7E%3A%7E%5Cmathbf%20Q%5E%5Ctop%5Cmathbf%20Q%3D%5Cmathbf%20I_d%7D%7B%5Ctext%7Bmax.%7D%7D%5Csum_%7Bn%3D1%7D%5EN%5Cleft%5C%7C%5Cmathbf%20Q%5E%5Ctop%5Cmathbf%20X_n%5Cmathbf%20Q%5Cright%5C%7C_1.)

---
IEEE Xplore: https://ieeexplore.ieee.org/document/8969097

---
**Examples**

First, let us create a D-by-D-by-N tensor --i.e., a collection of N matrix measurements. 
```python
import algorithms as arescal
import numpy as np
import matplotlib.pyplot as plt
import imagesc as imagesc
D = 10
N = 10
tensor = np.random.randn(D, D, N)
```
A-RESCAL can be solved exactly for the special case d=1 with the following code:
```python
qopt, bopt, vopt = arescal.exact(tensor)
print('Metric attained by exact solution:\t\t' + str(vopt))
```
The solution to A-RESCAL can be efficiently approximated for the special case d=1 with the following piece of code:
```python
q, b, evo = arescal.efficient(tensor) 
print('Metric attained by approximate solution:\t' + str(evo[-1]))
```
Next, with the following code we can compare the metric attained by the exact solution with the metric evolution of the approximate algorithm.
```python
plt.plot(range(len(evo)), evo, label = 'Approximate solution')
plt.plot([1, len(evo)], [vopt, vopt], label = 'Exact solution')
plt.ylabel('Metric')
plt.xlabel('Iteration index')
plt.legend()
plt.show()
```
Last, we approximate a solution to A-RESCAL for any d less than D.
```python
d = 3
Q = arescal.efficient_deflation(tensor, d) 
```

---
**Questions/issues**
Inquiries regarding the scripts provided below are cordially welcome. In case you spot a bug, please let me know. 

---
**Citing**

If you use our algorithms, please cite [[1]](https://ieeexplore.ieee.org/document/8969097).

```bibtex

@INPROCEEDINGS{ARESCAL,
  author={D. G. {Chachlakis} and Y. {Tsitsikas} and E. E. {Papalexakis} and P. P. {Markopoulos}},
  booktitle={2019 IEEE Global Conference on Signal and Information Processing (GlobalSIP)}, 
  title={Robust Multi-Relational Learning With Absolute Projection Rescal}, 
  year={2019},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/GlobalSIP45357.2019.8969097}}
```
[[1]](https://ieeexplore.ieee.org/document/8969097) D. G. Chachlakis, Y. Tsitsikas, E. E. Papalexakis and P. P. Markopoulos, "Robust Multi-Relational Learning With Absolute Projection Rescal," 2019 IEEE Global Conference on Signal and Information Processing (GlobalSIP), Ottawa, ON, Canada, 2019, pp. 1-5, doi: 10.1109/GlobalSIP45357.2019.8969097.

---