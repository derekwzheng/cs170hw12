import csv
import numpy
import numpy.linalg as LA
from collections import Counter

# Global variable to specify the k value for the k nearest neighbors algorithm
K = 100

def readCSVFile(featuresFileName, labelsFileName):
    """ Read FILENAME.csv and LABELSFILENAME.csv. Turn them into a list of
        tuples and a list of integers, resplectively. Every element in the
        list is a tuple of 784-dimensional object. Every component of a tuple
        is a real number in [0, 1].
        Return two lists: trainingFeaturesList and trainingLabelsList.
    """
    trainingFeaturesList = []
    with open(featuresFileName + '.csv', newline='') as csvfile:
        digitReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in digitReader:
            lst = []
            for component in row:
                lst.append(float(component))
            trainingFeaturesList.append(tuple(lst))
        trainingLabelsList = []
    with open(labelsFileName + '.csv', newline='') as csvfile:
        digitReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in digitReader:
            for label in row:
                trainingLabelsList.append(int(label))
    return trainingFeaturesList, trainingLabelsList


def distance(t1, t2):
    """ calculate the euclidean distance between two tuples using
    numpy.linalg.norm, return as float """
    t1 = numpy.array(t1)
    t2 = numpy.array(t2)
    return LA.norm(t1-t2)

def KNN(k, featureList, labelList, image):
    """ Find the closest digit using k-nearest neigbor algorithm
    """
    distArray = []
    for item in featureList:
        dist = distance(item, image)
        distArray.append(dist)
    distArray = numpy.array(distArray)
    minArray = []
    for i in range(0, k):
        minIndex = numpy.argmin(distArray)
        minArray.append(minIndex)
        distArray[minIndex] = float("inf")
    for i in range(0, k):
        minArray[i] = labelList[minArray[i]]
    counts = Counter(minArray)
    return counts.most_common(1)[0][0]


def readValidationFeaturesFile(valFeaturesFileName):
    """ Read validation features file and return a list of tuples. """
    validationFeaturesList = []
    with open(valFeaturesFileName + '.csv', newline='') as csvfile:
        digitReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in digitReader:
            lst = []
            for component in row:
                lst.append(float(component))
            validationFeaturesList.append(tuple(lst))
    return validationFeaturesList

def main():
    fetureList, labelList \
        = readCSVFile("hw12data/digitsDataset/trainFeatures", \
                      "hw12data/digitsDataset/trainLabels")
    validationFeaturesList = readValidationFeaturesFile("hw12data/digitsDataset/valFeatures")
    for image in validationFeaturesList:
        print(KNN(K, fetureList, labelList, image))

if __name__ == "__main__":
    main()