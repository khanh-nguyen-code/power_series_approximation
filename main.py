import numpy as np
import matplotlib.pyplot as plt
from polynomial import InnerProdSubspace

np.seterr(all="raise")

s = InnerProdSubspace(domain=(-np.pi, +np.pi))
s.add_dim()
s.add_dim()
s.add_dim()
s.add_dim()
s.add_dim()


for e in s.basis:
    print(e)

f = np.sin
sin_poly = s.project_trapz(f, 0.000001)
print(sin_poly)
x = np.arange(-np.pi, +np.pi, 0.1)
y1 = f(x)
y2 = sin_poly(x)
plt.plot(x, y1, "r")
plt.plot(x, y2, "b")
plt.ylim(-2, +2)
plt.show()