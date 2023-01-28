import numpy as np


class kNN:

    def __init__(self, k, metric):
        self.__k = k
        self.__metric = metric

    def predict(self, train, image):
        neighbours = np.zeros((self.__k, 2)) + 10e10  # (расстояние, метка)
        for sample in train:
            dist = self.__metric(image, sample)
            if dist < np.max(neighbours[:, 0]):
                neighbours[np.argmax(neighbours[:, 0])] = np.array([dist, sample.get_label()])
        dist_label_pair = np.unique(neighbours[:, 1], return_counts=True)
        return int(dist_label_pair[0][np.argmax(dist_label_pair[1])])

    def predict_all(self, train, images):
        labels = np.zeros(len(images))
        for i in range(len(images)):
            labels[i] = self.predict(train, images[i])
        return labels
