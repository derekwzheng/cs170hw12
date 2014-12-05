import csv
import sys

def readFiles(correctLabelsFileName, outputLabelsFileName):
    correctLabelsList, outputLabelsList = [], []
    with open(correctLabelsFileName + '.csv', newline='') as csvfile:
        digitReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in digitReader:
            for label in row:
                correctLabelsList.append(int(label))
    with open(outputLabelsFileName + '.csv', newline='') as csvfile:
        digitReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in digitReader:
            for label in row:
                outputLabelsList.append(int(label))
    return correctLabelsList, outputLabelsList


def rates(k):
    correctLabelsList, outputLabelsList = readFiles("hw12data/digitsDataset/valLabels", "code/digitsOutput" + str(k))
    length = len(correctLabelsList)
    same = 0
    for i in range(length):
        if correctLabelsList[i] == outputLabelsList[i]:
            same += 1
    print("correct rate: " + str(same / length * 100) + "%")
    print("error rate: " + str((1 - same / length) * 100) + "%")
k = sys.argv[1]
rates(k)
