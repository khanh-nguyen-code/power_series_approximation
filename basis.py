from typing import Callable, List, Tuple, Iterable

import numpy as np
from numpy.polynomial import Polynomial, polynomial as P


def count_iter() -> Iterable[int]:
    i = 0
    yield i
    while True:
        i += 1
        yield i


class Basis:
    """
    class represents an orthogonal basis of the real inner product space of polynomials with respect to
    the inner product \\int_{a}^{b} f(x)g(x)dx where [a, b] is the interval domain.
    """
    vector_list: List[Polynomial]
    domain: Tuple[float, float]

    def __init__(self, domain: Tuple[float, float] = (-1, +1)):
        self.vector_list = []
        self.domain = domain

    def inner_product(self, poly1: Polynomial, poly2: Polynomial) -> float:
        integral = Polynomial(P.polyint((poly1 * poly2).coef))
        return integral(self.domain[1]) - integral(self.domain[0])

    def add_vector(self):
        """
        add another vector to the basis
        """
        raise NotImplemented

    def project(self, f: Callable[[np.ndarray, ], np.ndarray], dx: float = 1e-3) -> Iterable[Polynomial]:
        """
        project a real function to the subspace integrating by composite trapezoidal rule
        https://en.wikipedia.org/wiki/Trapezoidal_rule
        yield the projected polynomials
        """
        poly = Polynomial(coef=[0, ])
        yield poly

        x_arr = np.append(np.arange(self.domain[0], self.domain[1], dx), [self.domain[1]])
        fx_arr = f(x_arr)

        for i in count_iter():
            while len(self.vector_list) <= i:
                self.add_vector()
            e = self.vector_list[i]
            y = fx_arr * e(x_arr)
            inner_prod = np.trapz(y=y, x=x_arr)
            poly += inner_prod * e
            yield poly


class GramSchmidt(Basis):
    """
    the orthogonal basis is generated using Gram-Schmidt Process on the set of vectors {1, x, x^2, x^3, x^4, ...}
    """

    def add_vector(self):
        poly = Polynomial(coef=[0 for _ in range(len(self.vector_list))] + [1, ])
        for e in self.vector_list:
            poly -= self.inner_product(poly, e) * e
        poly /= np.sqrt(self.inner_product(poly, poly))
        self.vector_list.append(poly)


class Chebyshev(Basis):
    def add_vector(self):
        if len(self.vector_list) < 2:
            self.vector_list = [Polynomial(coef=[1, ]), Polynomial(coef=[0, 1])]
            return
        self.vector_list.append(2 * self.vector_list[1] * self.vector_list[-1] - self.vector_list[-2])
