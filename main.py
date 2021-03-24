import numpy as np
import matplotlib.pyplot as plt
from polynomial import InnerProdSubspace

np.seterr(all="raise")

s = InnerProdSubspace(domain=(-np.pi, +np.pi))
f = np.sin

for dim, sin_approx in enumerate(s.project(f, 0.001)):
    if dim > 5:
        break
    x = np.arange(-np.pi, +np.pi, 0.1)
    y1 = f(x)
    y2 = sin_approx(x)
    plt.plot(x, y1, "r")
    plt.plot(x, y2, "b")
    plt.ylim(-2, +2)
    plt.title(f"dim {dim}")
    print(f"dim {dim}: {sin_approx}")
    plt.show()
