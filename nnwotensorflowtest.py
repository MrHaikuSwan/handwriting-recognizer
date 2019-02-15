import nnwotensorflow
from tensorflow import keras
import time


mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#testimg = test_images[0]
#testlabel = test_labels[0]
#
#model = nnwotensorflow.Model([784,128,128,10], 1.0)
#starttime = time.time()
#
#model.train(train_images, train_labels, batch_size=100, epochs=5)
#print(time.time()-starttime)

