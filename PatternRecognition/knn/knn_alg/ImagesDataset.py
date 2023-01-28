import numpy as np

from knn.image.Pattern import Pattern


class ImagesDataset:

    def __init__(self, N_numbers, img_type):
        self.__img_type = img_type
        self.__N_numbers = N_numbers
        self.__data = None

    def initialize(self):
        data = []
        for pattern in Pattern:
            for i in range(self.__N_numbers):
                data.append(self.__img_type.value(pattern.pixels(), pattern.label()).get_noised_img(np.random.random()))
        self.__data = np.array(data)

    def initialize(self, proba):
        data = []
        for pattern in Pattern:
            for i in range(self.__N_numbers):
                data.append(self.__img_type.value(pattern.pixels(), pattern.label()).get_noised_img(proba))
        self.__data = np.array(data)

    def create_img(self, pattern, proba):
        return self.__img_type.value(pattern.pixels(), pattern.label()).get_noised_img(proba)

    def get_targets(self):
        res = []
        for sample in self.__data:
            res.append(sample.get_label())
        return np.array(res)

    def __getitem__(self, key):
        return self.__data[key]

    def __len__(self):
        return len(self.__data)
