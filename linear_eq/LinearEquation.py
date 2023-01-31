class LinearEquation:

    def __init__(self, coefs, f, start_conditions, a, b):
        self.__coefs = coefs
        self.__f = f
        self.__start = start_conditions
        self.__a = a
        self.__b = b
        self.__h = b - a

    # Решение явным методом Рунге-Кутты 4-го порядка
    def solve_by_explicit_method(self, N):
        pass

    # Решение неявным методом Рунге-Кутты 2-го порядка
    def solve_by_implicit_method(self, N):
        pass

    def get_coefs(self):
        return self.__coefs.copy()

    def get_h(self, N):
        self.__h /= (N - 1)
        return self.__h

    def get_dim(self):
        return len(self.__start)

    def get_start_cond(self):
        return self.__start.copy()

    def get_f(self, t):
        return self.__f(t)

    # Получение координаты t для ТЕКУЩЕГО шага, то есть для того, на котором считается ноевое значение
    def get_t_n(self, n):
        return self.__a + n*self.__h
