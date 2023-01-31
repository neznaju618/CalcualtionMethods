import numpy as np

from sciproj.linear_eq.LinearEquation import LinearEquation


class LinearEquationWithConstantCoefs(LinearEquation):

    def __init__(self, coefs, f, start_conditions, a, b):
        super().__init__(coefs, f, start_conditions, a, b)

    def solve_by_explicit_method(self, N):
        h = self.get_h(N)
        coefs = self.get_coefs()
        dim = self.get_dim()
        result = np.zeros((dim, N))  # k-ая строка в result соответсвтует k-ой производной решения.
        result[:, 0] = self.get_start_cond()
        F = np.zeros(dim)
        F[:dim - 1] = result[1:, 0]
        for n in range(1, N):
            F[dim - 1] = (self.get_f(self.get_t_n(n) - h) - np.dot(coefs[:dim], result[:, n - 1])) / coefs[dim]
            # k1 = F
            k2 = F.copy()
            k2[:dim - 1] += h / 2 * F[:dim - 1]
            k2[dim - 1] = self.get_f(self.get_t_n(n) - h / 2) - np.dot(coefs[:dim],
                                                                       result[:dim, n - 1] + h / 2 * F[:dim])

            k3 = F.copy()
            k3[:dim - 1] += h / 2 * k2[:dim - 1]
            k3[dim - 1] = self.get_f(self.get_t_n(n) - h / 2) - np.dot(coefs[:dim],
                                                                       result[:dim, n - 1] + h / 2 * k2[:dim])

            k4 = F.copy()
            k3[:dim - 1] += h * k3[:dim - 1]
            k3[dim - 1] = self.get_f(self.get_t_n(n)) - np.dot(coefs[:dim],
                                                               result[:, n - 1] + h / 2 * k3[:dim])
            result[:, n] = result[:, n - 1] + h / 6 * (F + 2 * k2 + 2 * k3 + k4)
            F[:dim - 1] = result[1:, n]
        return result

    def solve_by_implicit_method(self, N):
        h = self.get_h(N)
        coefs = self.get_coefs()
        dim = self.get_dim()
        result = np.zeros((dim, N))  # k-ая строка в result соответсвтует k-ой производной решения.
        result[:, 0] = self.get_start_cond()
        F = np.zeros(dim)
        F[:dim - 1] = result[1:, 0]
        for n in range(1, N):
            F[dim - 1] = (self.get_f(self.get_t_n(n) - h) - np.dot(coefs[:dim], result[:, n - 1])) / coefs[dim]
            forecast = result[:, n - 1] + h * F
            F_n = np.zeros(dim)
            F_n[:dim-1] = forecast[1:]
            F_n[dim-1] = (self.get_f(self.get_t_n(n) - h) - np.dot(coefs[:dim], forecast)) / coefs[dim]
            result[:, n] = result[:, n - 1] + h * (F + F_n) / 2
            F[:dim - 1] = result[1:, n]
        return result

