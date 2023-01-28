import numpy as np

from knn.image.Image import Image


class GrayGradationsImage(Image):

    def __init__(self, pixels, label):
        if np.max(pixels) == 1:
            super().__init__(pixels * 255, label)
        else:
            super().__init__(pixels, label)

    def get_noised_img(self, proba):
        center = np.floor(np.random.random(size=2) * 8)
        length = np.floor(np.sqrt(proba) * 8)
        noise = np.zeros((8, 8))
        for i in range(int(length)):
            for j in range(int(length)):
                noise[(int(center[0]) + i) % 8][(int(center[1]) + j) % 8] = np.floor(np.random.random() * 255)
        return GrayGradationsImage(np.mod(self.get_pixels().dot(np.eye(8, 8,
                                                                       int(np.round(
                                                                           proba * (-4 + np.random.random() * 8))))) +
                                          noise, 256), self.get_label())
