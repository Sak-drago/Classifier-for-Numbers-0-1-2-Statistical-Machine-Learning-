import numpy as np
from os.path import join

data = np.load("/home/sakie/Desktop/repos/Classifier-for-Numbers-0-1-2-Statistical-Machine-Learning-/train_data/dataSet.npz")
flattenImages = data["images"]
labels = data["labels"]

print(flattenImages.shape)
means = np.zeros((3,784), dtype = np.float64)


for idx in range(300):
    if labels[idx] == 0:
        means[0] += flattenImages[idx]
    elif labels[idx] == 1:
        means[1] += flattenImages[idx]
    elif labels[idx] == 2:
        means[2] += flattenImages[idx]

means = means/100.0
centralisedMatrix = [[], [], []]
for idx in range(300):
    if labels[idx] == 0:
        centralisedMatrix[0].append(flattenImages[idx] - means[0])
    elif labels[idx] == 1:
        centralisedMatrix[1].append(flattenImages[idx] - means[1])
    elif labels[idx] == 2:
        centralisedMatrix[2].append(flattenImages[idx] - means[2])


covarMatrix = []
for cl in range(3):
    centralisedMatrixPerClass = np.array(centralisedMatrix[cl])
    covarMatrixPerClass = np.dot(centralisedMatrixPerClass.transpose(), centralisedMatrixPerClass) / (100-1)
    covarMatrix.append(covarMatrixPerClass)
     
for row in range(784):
    for col in range(784):
        if covarMatrix[0][row][col]>0:
            print(covarMatrix[0][row][col])
