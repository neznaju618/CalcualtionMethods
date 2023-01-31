class Function:

    def __init__(self, function):
        self.__function = function

    def __call__(self, t, y):
        return self.__function(t, y)
