import numpy as np
import matplotlib.pyplot as plt
from polynomial import InnerProductSpace

if __name__ == "__main__":
    np.seterr(all="raise")

    domain = (-4*np.pi, +4*np.pi)

    s = InnerProductSpace(domain=domain)
    f = np.sin

    for dim, sin_approx in enumerate(s.project(f, 0.001)):
        x = np.arange(domain[0], domain[1], 0.1)
        y1 = f(x)
        y2 = sin_approx(x)
        plt.plot(x, y1, "r")
        plt.plot(x, y2, "b")
        plt.ylim(-2, +2)
        plt.title(f"dim {dim}")
        print(f"dim {dim}: {sin_approx}")
        plt.show()
