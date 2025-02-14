import numpy as np
from os.path import join
from scipy.linalg import eigh

data = np.load("/home/sakie/Desktop/repos/Classifier-for-Numbers-0-1-2-Statistical-Machine-Learning-/train_data/dataSet.npz")
flattenImages = data["images"]
labels = data["labels"]

test = np.load("/home/sakie/Desktop/repos/Classifier-for-Numbers-0-1-2-Statistical-Machine-Learning-/train_data/testSet.npz")
testImages = test["images"]
testLabels = test["labels"]

priors = [1/3,1/3,1/3]
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

def CalculateMeanAndCovariance(DataMatrix):
    nSamples = DataMatrix.shape[0]
    # Calculate mean vector (average across samples)
    MeanVec = np.mean(DataMatrix, axis=0)
    # Center the data by subtracting the mean
    CenteredData = DataMatrix - MeanVec
    # Compute covariance matrix: (X_centered^T * X_centered) / (nSamples - 1)
    CovMatrix = np.dot(CenteredData.T, CenteredData) / (nSamples - 1)
    return MeanVec, CovMatrix
# =======================================================================================================


# def multiVarGaussian(data, mean, coVarMatrix):
#     balancingFactor = 1e-3      
#     coVarMatrix_reg = coVarMatrix + balancingFactor * np.eye(coVarMatrix.shape[0])
    
#     sign, logDet = np.linalg.slogdet(coVarMatrix_reg)
#     if sign <= 0:
#         raise ValueError("Covariance matrix is not positive definite.")
    
#     invCov = np.linalg.inv(coVarMatrix_reg)
#     d = data.shape[0]  
#     diffVec = data - mean
    
#     exponentTerm = -0.5 * np.dot(diffVec.T, np.dot(invCov, diffVec))
#     logNormalization = (d / 2) * np.log(2 * np.pi) + 0.5 * logDet
#     logProb = exponentTerm - logNormalization   
#     return logProb

# def MLEClassifier(x, Means, CovarMatrices):
#     Likelihoods = []
#     for c in range(len(Means)):
#         likelihood = multiVarGaussian(x, Means[c], CovarMatrices[c])
#         Likelihoods.append(likelihood)
#     return np.argmax(Likelihoods)

# xTest = testImages
# yTest = testLabels
# CorrectCount = 0
# Predictions = []
# numTestSamples = len(xTest)  # Number of test samples

# for i in range(numTestSamples):
#     predLabel = MLEClassifier(xTest[i], means, covarMatrix)
#     Predictions.append(predLabel)
#     if predLabel == yTest[i]:
#         CorrectCount += 1

# Accuracy = (CorrectCount / numTestSamples) * 100
# print(f"MLE Test Accuracy: {Accuracy:.2f}%")
# Accidentally did RL LMAO
# =======================================================================================================

def principalComponentAnalysisGlobal(covarMatrix, centralisedMatrix, dim, xTest):
    eigenValues, eigenVectors = np.linalg.eigh(covarMatrix)
    sortedIndex = np.argsort(eigenValues)[::-1]  
    sortedEigenValues = eigenValues[sortedIndex]
    sortedEigenVectors = eigenVectors[:, sortedIndex]

    Up = sortedEigenVectors[:, :dim]

    totalVar = np.sum(eigenValues)
    varExplained = np.sum(sortedEigenValues[:dim]) / totalVar
    Y = np.dot(testImages, Up)
    print(f"Variance explained by {dim} dimensions: {varExplained:.2f}")
    return Y, Up

globalCovMatrix = np.dot(genMean.transpose(),genMean) / 299;

dim = 300 # Dimensionality to reduce to
Y, Up = principalComponentAnalysisGlobal(globalCovMatrix, genMean, dim, testImages)


print("Global PCA COMPLETE")

# =======================================================================================================

#Raw implementation of FDA
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

def performFda(X, y, genMean):  # FDA function
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
    return W

# =======================================================================================================

W = performFda(means, labels, genMean)
Y = np.dot(testImages, W)

meanY , covY = CalculateMeanAndCovariance(Y) 
accuracyFdaLda = accuracyFunction(ldaClassifier,Y,testLabels, meanY, covY, priors)
print(f"FDA + LDA Test Accuracy: {accuracyFdaLda:.2f}%")
