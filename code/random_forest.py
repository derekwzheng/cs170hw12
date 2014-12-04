import csv
import numpy
import numpy.linalg as LA
from collections import Counter
import random
from math import log2

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

def countSpamAndHam(labelsList):
    cnt = Counter()
    for label in labelsList:
        cnt[label] += 1
    return cnt


# Global variables to store the training set
featuresList, labelsList \
    = readCSVFile("../hw12data/emailDataset/trainFeatures", \
                      "../hw12data/emailDataset/trainLabels")
trainingDic = dict(zip(featuresList, labelsList))
# Global Constants
FEATURES = list(range(0, 57))
NUMBER_SELECTED_FEATURES = 8
SPAM = 1
NONSPAM = 0
TRAINING_COUNTER = countSpamAndHam(labelsList)

def entropy(s):
    labels = [trainingDic[email] for email in s]
    cnt = countSpamAndHam(labels)
    return -((cnt[SPAM]) / len(labels)) * log2((cnt[SPAM]) / len(labels)) \
        - ((cnt[NONSPAM]) / len(labels)) * log2(cnt[NONSPAM] / len(labels))

class LeafNode:
    def __init__(self, classification):
        self.classification = classification

class InternalNode:
    def __init__(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

def isSameClass(lst):
    length = len(lst)
    for i in range(length - 1):
        if trainingDic[lst[i]] != trainingDic[lst[i + 1]]:
            return False
    return True

def split(lst, feature, threshold):
    leftList, rightList = [], []
    for email in lst:
        if email[feature] <= threshold:
            leftList.append(email)
        else:
            rightList.append(email)
    return leftList, rightList

def informationGain(s, sLeft, sRight):
    return entropy(s) - ((len(sLeft) / len(s) * entropy(sLeft)) + len(sRight) \
                         / len(s) * entropy(sRight))

def build_tree(s):
    """ It take S, a list of features vectors. """
    if isSameClass(s):
        return LeafNode(trainingDic[s[0]])
    argmaxF, argmaxT, maxInfoGain = None, None, -float("inf")
    for f in random.sample(FEATURES, NUMBER_SELECTED_FEATURES):
        values = {featuresList[x[f]] for x in featuresList}
        for t in values:
            sLeft, sRight = split(s, f, t)
            infoGain = informationGain(s, sLeft, sRight)
            if infoGain > maxInfoGain:
                argmaxF, argmaxT = f, t
                maxInfoGain = infoGain
    sLeft, sRight = split(s, argmaxF, argmaxT)
    return InternalNode(argmaxF, argmaxT, build_tree(sLeft), build_tree(sRight))

def main():
    fetureList, labelList \
        = readCSVFile("../hw12data/emailDataset/trainFeatures", \
                      "../hw12data/emailDataset/trainLabels")
    validationFeaturesList = readValidationFeaturesFile("../hw12data/emailDataset/valFeatures")
    for image in validationFeaturesList:
        # FIXME

if __name__ == "__main__":
    main()