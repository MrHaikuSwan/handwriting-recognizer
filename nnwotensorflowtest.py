from tensorflow import keras
import time

import nnwotensorflow

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = nnwotensorflow.Model([784,128,10], 1.0)

starttime = time.time()
model.train(train_images, train_labels, batch_size=100, epochs=10)
elapsedtime = time.time()-starttime

trainset_performance = model.test(train_images, train_labels)[1]
testset_performance = model.test(test_images, test_labels)[1]

print("Elapsed Time:", elapsedtime)
print("Training Set Performance: {}%".format(trainset_performance))
print("Testing Set Performance: {}%".format(testset_performance))