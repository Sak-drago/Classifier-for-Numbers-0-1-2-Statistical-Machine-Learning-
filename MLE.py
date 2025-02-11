import numpy as np
from os.path import join

data = np.load("/home/sakie/Desktop/repos/Classifier-for-Numbers-0-1-2-Statistical-Machine-Learning-/train_data/dataSet.npz")
flattenImages = data["images"]
labels = data["labels"]

print(flattenImages.shape)
print("Max pixel value:", np.max(flattenImages))
means = np.zeros((3,784), dtype = np.float64)

print(flattenImages)

for idx in range(300):
    if labels[idx] == 0:
        means[0] += flattenImages[idx]
    elif labels[idx] == 1:
        means[1] += flattenImages[idx]
    elif labels[idx] == 2:
        means[2] += flattenImages[idx]

means = means/100.0

