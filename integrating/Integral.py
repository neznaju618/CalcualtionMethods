class Integral:

    def __init__(self, function, region):
        self.__f = function
        self.__R = region

    def integrate_by_MonteCarlo(self, N):
        points = self.__R.generate_points(N)
        result = 0
        for i in range(N):
            result += self.__f(points[i])
        return result / N * self.__R.get_measure()
