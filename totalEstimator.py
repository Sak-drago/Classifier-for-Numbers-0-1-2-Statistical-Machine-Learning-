import numpy as np
from os.path import join
from scipy.linalg import eigh

data = np.load("/home/sakie/Desktop/repos/Classifier-for-Numbers-0-1-2-Statistical-Machine-Learning-/train_data/dataSet.npz")
flattenImages = data["images"]
labels = data["labels"]

test = np.load("/home/sakie/Desktop/repos/Classifier-for-Numbers-0-1-2-Statistical-Machine-Learning-/train_data/testSet.npz")
testImages = test["images"]
testLabels = test["labels"]

means = np.zeros((3,784), dtype = np.float64)
genMean = np.zeros((1,784), dtype = np.float64)

for idx in range(300):
    if labels[idx] == 0:
        means[0] += flattenImages[idx]
    elif labels[idx] == 1:
        means[1] += flattenImages[idx]
    elif labels[idx] == 2:
        means[2] += flattenImages[idx]
    genMean += flattenImages[idx]

means = means/100.0
genMean = genMean/300.0
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

# =======================================================================================================


def multiVarGaussian(data, mean, coVarMatrix):
    balancingFactor = 1e-3      
    coVarMatrix_reg = coVarMatrix + balancingFactor * np.eye(coVarMatrix.shape[0])
    
    sign, logDet = np.linalg.slogdet(coVarMatrix_reg)
    if sign <= 0:
        raise ValueError("Covariance matrix is not positive definite.")
    
    invCov = np.linalg.inv(coVarMatrix_reg)
    d = data.shape[0]  
    diffVec = data - mean
    
    exponentTerm = -0.5 * np.dot(diffVec.T, np.dot(invCov, diffVec))
    logNormalization = (d / 2) * np.log(2 * np.pi) + 0.5 * logDet
    logProb = exponentTerm - logNormalization   
    return logProb

def MLEClassifier(x, Means, CovarMatrices):
    Likelihoods = []
    for c in range(len(Means)):
        likelihood = multiVarGaussian(x, Means[c], CovarMatrices[c])
        Likelihoods.append(likelihood)
    return np.argmax(Likelihoods)

xTest = testImages
yTest = testLabels
CorrectCount = 0
Predictions = []
numTestSamples = len(xTest)  # Number of test samples

for i in range(numTestSamples):
    predLabel = MLEClassifier(xTest[i], means, covarMatrix)
    Predictions.append(predLabel)
    if predLabel == yTest[i]:
        CorrectCount += 1

Accuracy = (CorrectCount / numTestSamples) * 100
print(f"MLE Test Accuracy: {Accuracy:.2f}%")
# =======================================================================================================

def principalComponentAnalysisGlobal(covarMatrix, centralisedMatrix, dim):
    eigenValues, eigenVectors = np.linalg.eigh(covarMatrix)
    sortedIndex = np.argsort(eigenValues)[::-1]  
    sortedEigenValues = eigenValues[sortedIndex]
    sortedEigenVectors = eigenVectors[:, sortedIndex]

    Up = sortedEigenVectors[:, :dim]

    totalVar = np.sum(eigenValues)
    varExplained = np.sum(sortedEigenValues[:dim]) / totalVar
    Y = np.dot(centralisedMatrix, Up)
    print(f"Variance explained by {dim} dimensions: {varExplained:.2f}")
    return Y, Up

allCentralizedData = np.concatenate(centralisedMatrix, axis=0)
globalCovMatrix = np.cov(allCentralizedData, rowvar=False)

dim = 80 # Dimensionality to reduce to
reducedMatrix, Up = principalComponentAnalysisGlobal(globalCovMatrix, allCentralizedData, dim)

print("Global PCA COMPLETE")

# =======================================================================================================

genMean = genMean.reshape(784,)
sB = np.zeros((784,784))
sW = np.zeros((784,784))

for cl in range(3):
    diff = means[cl]-genMean
    diff = diff.reshape(784,)
    sB+= 100*(np.dot(diff,diff.transpose()))

    xC = np.array(centralisedMatrix[cl]).reshape(784,100)
    sW+= np.dot(xC, xC.transpose())

sW += 1e-6 * np.eye(sW.shape[0])
eigenValues, eigenVectors = eigh(sB,sW)
sortedIndex = np.argsort(eigenValues)[::-1]
eigenValues = eigenValues[sortedIndex]
eigenVectors = eigenVectors[:, sortedIndex]

W = eigenVectors[:, :2]
# =======================================================================================================

def ldaClassifier(x, means, pooled_cov, priors):
    inv_pooled_cov = np.linalg.inv(pooled_cov)
    scores = []
    for c in range(len(means)):
        score = np.dot(x, np.dot(inv_pooled_cov, means[c])) - 0.5 * np.dot(means[c].T, np.dot(inv_pooled_cov, means[c])) + np.log(priors[c])
        scores.append(score)
    return np.argmax(scores)

def qdaClassifier(x, means, covariances, priors):
    scores = []
    for c in range(len(means)):
        cov_reg = covariances[c] + 1e-6 * np.eye(covariances[c].shape[0])
        inv_cov = np.linalg.inv(cov_reg)
        sign, logdet = np.linalg.slogdet(cov_reg)
        diff = x - means[c]
        score = -0.5 * logdet - 0.5 * np.dot(diff.T, np.dot(inv_cov, diff)) + np.log(priors[c])
        scores.append(score)
    return np.argmax(scores)

def accuracyFunction(classifier, data, labels, *args):
    correct_predictions = 0
    for i in range(len(data)):
        prediction = classifier(data[i], *args)
        if prediction == labels[i]:
            correct_predictions += 1
    accuracy = (correct_predictions / len(data)) * 100
    return accuracy

def performFda(X, y, dim):  # FDA function
    nSamplesPerClass = [np.sum(y == c) for c in np.unique(y)]
    means = [np.mean(X[y == c], axis=0) for c in np.unique(y)]
    genMean = np.mean(X, axis=0)

    sB = np.zeros((X.shape[1], X.shape[1]))
    sW = np.zeros((X.shape[1], X.shape[1]))

    for cl in range(len(np.unique(y))):
        diff = means[cl] - genMean
        sB += nSamplesPerClass[cl] * np.dot(diff.reshape(-1, 1), diff.reshape(1, -1))

        class_data = X[y == np.unique(y)[cl]]
        sW += np.dot(class_data.T, class_data)  # Correct sW calculation

    sW += 1e-6 * np.eye(sW.shape[1])
    eigenValues, eigenVectors = eigh(sB, sW)
    sortedIndex = np.argsort(eigenValues)[::-1]
    eigenVectors = eigenVectors[:, sortedIndex]

    W = eigenVectors[:, :dim]
    return W

# =======================================================================================================
pooled_cov = np.zeros((784, 784))
for c in range(3):
    pooled_cov += (covarMatrix[c] * (100 - 1))

pooled_cov /= (300 - 3)
pooled_cov += 1e-6 * np.eye(pooled_cov.shape[0]) 
priors = np.array([1/3, 1/3, 1/3])
#lda_accuracy = accuracy_function(lda_classifier, flattenImages, labels, means, pooled_cov, priors)
#qda_accuracy = accuracy_function(qda_classifier, flattenImages, labels, means, covarMatrix, priors)
#print(f"LDA Classifier Accuracy: {lda_accuracy:.2f}%")
#print(f"QDA Classifier Accuracy: {qda_accuracy:.2f}%")

# =======================================================================================================
trainLabel = labels[:300]
UniqueLabels = np.unique(trainLabel)
NumClasses = len(UniqueLabels)
ReducedMeans = np.zeros((NumClasses, dim))
for i, label in enumerate(UniqueLabels):
    ClassData = reducedMatrix[trainLabel == label]  # All samples for a given class
    ReducedMeans[i] = np.mean(ClassData, axis=0)           # (Dim,)

reducedCovariances = []
PooledCovariance = np.zeros((dim, dim))
for i, label in enumerate(UniqueLabels):
    ClassData = reducedMatrix[trainLabel== label]
    ClassCovariance = np.cov(ClassData, rowvar=False)
    reducedCovariances.append(ClassCovariance)
    PooledCovariance += ClassCovariance * (ClassData.shape[0] - 1)
PooledCovariance /= (reducedMatrix.shape[0] - NumClasses)
PooledCovariance += 1e-6 * np.eye(dim)

Priors = np.array([1/NumClasses] * NumClasses)

predictionsLDA = []
preedictionsQDA = []
#for i in range(reducedMatrix.shape[0]):
#    pred = ldaClassifier(reducedMatrix[i], ReducedMeans, PooledCovariance, Priors)
#    predictionsLDA.append(pred)
#    pred = qdaClassifier(reducedMatrix[i], ReducedMeans, reducedCovariances, Priors)
#    preedictionsQDA.append(pred)


#Predictions = np.array(predictionsLDA)
#Accuracy = np.mean(Predictions == trainLabel) * 100
#print(f"LDA Accuracy on training data after global PCA: {Accuracy:.2f}%")

#Predictions = np.array(preedictionsQDA)
#Accuracy = np.mean(Predictions == trainLabel) * 100

W = performFda(allCentralizedData, labels, dim)  # Use all data for FDA

reducedAllData = np.dot(allCentralizedData, W)
meansFda = [np.mean(reducedAllData[labels == c], axis=0) for c in np.unique(labels)]
pooledCovFda = np.cov(reducedAllData, rowvar=False) + 1e-6 * np.eye(dim)
priorsFda = np.array([np.sum(labels == c) / len(labels) for c in np.unique(labels)])

numIterations = 1  # Number of random test sets
fdaAccuracies = []

for _ in range(numIterations):
    randomIndices = np.random.choice(300, 100, replace=False)
    XTest = testImages[randomIndices]
    yTest = testLabels[randomIndices]

    reducedMatrixFdaTest = np.dot(XTest, W)

    ldaAccuracyFda = accuracyFunction(ldaClassifier, reducedMatrixFdaTest, yTest, meansFda, pooledCovFda, priorsFda)
    fdaAccuracies.append(ldaAccuracyFda)

averageFdaAccuracy = np.mean(fdaAccuracies)
print(f"Average FDA accuracy over {numIterations} test sets: {averageFdaAccuracy:.2f}%")
# =======================================================================================================
