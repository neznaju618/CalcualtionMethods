import numpy as np
import matplotlib.pyplot as plt

from knn.image.ImageType import ImageType
from knn.image.Pattern import Pattern
from knn.knn_alg.ImagesDataset import ImagesDataset
from mnk.mnk_alg.MNKDataset import MNKDataset
from mnk.mnk_alg.Plotter import plot, plot_NN
from neuralnetwork.ActivationFunctions import ActivationFunctions
from neuralnetwork.ErrorFunctions import ErrorFunctions
from neuralnetwork.LayerInfo import LayerInfo
from neuralnetwork.NeuralNetwork import NeuralNetwork

layers_info = [LayerInfo(ActivationFunctions.TANH, 64),
               LayerInfo(ActivationFunctions.ReLU, 64),
               LayerInfo(ActivationFunctions.TANH, 64),
               LayerInfo(ActivationFunctions.ReLU, 64),
               LayerInfo(ActivationFunctions.TANH, 64),
               LayerInfo(ActivationFunctions.SOFTMAX, 10)]
NN = NeuralNetwork(layers_info, 64)

train = ImagesDataset(100, ImageType.MONOCHROME)
train.initialize(0.2)

test = train.create_img(Pattern.FIVE, 0.2)

image_pixels = []
answers = []

for i in range(len(train)):
    image_pixels.append(train[i].get_pixels().ravel())
    answer = np.zeros(10)
    answer[train[i].get_label()] = 1
    answers.append(answer)

NN.fit_by_dataset(image_pixels, answers, ErrorFunctions.LOG, 0.0001, epochs=5000)
print(NN.output(test.get_pixels().ravel()))
