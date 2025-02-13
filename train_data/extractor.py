from array import array
import numpy as np
import struct
from os.path import join
import random
import matplotlib.pyplot as plt

class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)

input_path = '../train_data/archive/'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1

dataSetImages = []
dataSetLabels = []
testSetImages = []
testSetLabels =[]
flattenDataSetImages = []
flattenTestSetImages = []

countZero = 0
countOne = 0
countTwo = 0

for i in range(60000):
    if countZero == 100 and countOne == 100 and countTwo == 100:
        break
    r = random.randint(0, 59999)
    if y_train[r] == 0 and countZero <100:
        countZero+=1
        dataSetImages.append(x_train[r])
        dataSetLabels.append(y_train[r])
        flattenImage = np.reshape(x_train[r], (784,)) / 255.0
        flattenDataSetImages.append(flattenImage)

    elif y_train[r] == 1 and countOne <100:
        countOne+=1
        dataSetImages.append(x_train[r])
        dataSetLabels.append(y_train[r])
        flattenImage = np.reshape(x_train[r], (784,)) / 255.0
        flattenDataSetImages.append(flattenImage)

    elif y_train[r] == 2 and countTwo <100:
        countTwo+=1
        dataSetImages.append(x_train[r])
        dataSetLabels.append(y_train[r])
        flattenImage = np.reshape(x_train[r], (784,)) / 255.0
        flattenDataSetImages.append(flattenImage)

show_images(dataSetImages, dataSetLabels)
print(len(dataSetLabels))
print(len(dataSetImages))

countZero = 0
countOne = 0
countTwo = 0
for i in range(15000):
    if countZero == 100 and countOne == 100 and countTwo == 100:
        break
    r = random.randint(0, 10000)
    if y_test[r] == 0 and countZero <100:
        countZero+=1
        testSetImages.append(x_test[r])
        testSetLabels.append(y_test[r])
        flattenImage = np.reshape(x_test[r], (784,)) / 255.0
        flattenTestSetImages.append(flattenImage)

    elif y_test[r] == 1 and countOne<100:
        countOne+=1
        testSetImages.append(x_test[r])
        testSetLabels.append(y_test[r])
        flattenImage = np.reshape(x_test[r], (784,)) / 255.0
        flattenTestSetImages.append(flattenImage)

    elif y_test[r] == 2 and countTwo<100:
        countTwo+=1
        testSetImages.append(x_test[r])
        testSetLabels.append(y_test[r])
        flattenImage = np.reshape(x_test[r], (784,)) / 255.0
        flattenTestSetImages.append(flattenImage)

np.savez('dataSet.npz', images=flattenDataSetImages, labels=dataSetLabels)
np.savez('testSet.npz', images=flattenTestSetImages, labels=testSetLabels)
print(len(testSetLabels))
print(len(testSetImages))
