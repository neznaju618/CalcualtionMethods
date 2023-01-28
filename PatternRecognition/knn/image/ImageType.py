from enum import Enum

from knn.image.GrayGradationsImage import GrayGradationsImage
from knn.image.MonochromeImage import MonochromeImage


class ImageType(Enum):
    MONOCHROME = MonochromeImage
    GRAY_GRADES = GrayGradationsImage

    def __init__(self, img_class):
        self.__img_class = img_class
