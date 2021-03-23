from typing import Callable

import numpy as np
from numpy.polynomial import Polynomial, polynomial as P


class InnerProdSubspace:
    """
    class represents an orthogonal basis with respect to \int_{a}^{b} f(x)g(x)dx where [a, b] is the domain
    generating using Gram-Schmidt Process on 1, x, x^2, x^3, ...
    """
    basis: list[Polynomial]
    domain: tuple[float, float]

    def __init__(self, domain: tuple[float, float] = (-1, +1)):
        self.basis = []
        self.domain = domain

    def dim(self) -> int:
        return len(self.basis)

    def inner_prod(self, poly1: Polynomial, poly2: Polynomial) -> float:
        integral = Polynomial(P.polyint((poly1 * poly2).coef))
        return integral(self.domain[1]) - integral(self.domain[0])

    def add_dim(self):
        """
        add another vector to the basis
        :return:
        """
        poly = Polynomial(
            coef=[0.0 for _ in range(self.dim())] + [1.0, ],
            # domain=(-np.inf, +np.inf),
            # window=(-np.inf, +np.inf),
        )
        for e in self.basis:
            poly -= self.inner_prod(poly, e) * e
        poly /= np.sqrt(self.inner_prod(poly, poly))
        self.basis.append(poly)

    def inner_prod_trapz(self, func1: Callable[[np.ndarray, ], np.ndarray], func2: Callable[[np.ndarray, ], np.ndarray],
                         dx: float = 1e-3) -> float:
        x = np.append(np.arange(self.domain[0], self.domain[1], dx), [self.domain[1]])
        y = func1(x) * func2(x)
        return np.trapz(y=y, x=x)

    def project_trapz(self, f: Callable[[np.ndarray, ], np.ndarray], dx: float = 1e-3) -> Polynomial:
        """
        project a real function to the subspace using the composite trapezoidal rule
        https://numpy.org/doc/stable/reference/generated/numpy.trapz.html#numpy.trapz
        :return: the projected Polynomial
        """
        poly = Polynomial(
            coef=[0.0, ],
            # domain=(-np.inf, +np.inf),
            # window=(-np.inf, +np.inf),
        )
        for e in self.basis:
            poly += self.inner_prod_trapz(f, e, dx) * e
        return poly