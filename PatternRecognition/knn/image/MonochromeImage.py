import numpy as np

from knn.image.Image import Image


class MonochromeImage(Image):

    def __init__(self, pixels, label):
        super().__init__(pixels, label)

    def shift_img(self, n):
        return MonochromeImage(self.get_pixels().dot(np.eye(8, 8, n)), self.get_label())

    def get_noised_img(self, proba):
        return MonochromeImage(np.mod(np.floor(np.random.random(size=(8, 8)) + proba) +
                                      self.get_pixels().dot(np.eye(8, 8,
                                                                   int(np.floor(
                                                                       proba*(-4 + np.random.random() * 8))))), 2),
                               self.get_label())

