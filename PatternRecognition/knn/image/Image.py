class Image:

    def __init__(self, pixels, label):
        self.__pixels = pixels
        self.__label = label

    def __getitem__(self, key):
        return self.__pixels[key]

    def __sub__(self, other):
        return self.__pixels - other.__pixels

    def get_label(self):
        return self.__label

    def get_pixels(self):
        return self.__pixels.copy()

    def get_noised_img(self, proba):
        pass
