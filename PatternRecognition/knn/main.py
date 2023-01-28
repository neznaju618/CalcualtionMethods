from knn.image.ImageType import ImageType
from knn.knn_alg.ImagesDataset import ImagesDataset
from knn.knn_alg.Metric import Metric
from knn.knn_alg.kNN import kNN
from knn.knn_alg.ImagesConfusionMatrix import ImagesConfusionMatrix

train = ImagesDataset(100, ImageType.MONOCHROME)
train.initialize(0.2)

test = ImagesDataset(25, ImageType.MONOCHROME)
test.initialize(0.2)

knn = kNN(5, Metric.d_2)
y_pred = knn.predict_all(train, test)
y = test.get_targets()

cm = ImagesConfusionMatrix(y, y_pred)
print(cm.show())
