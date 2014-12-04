import csv

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


def correctRate():
    correctLabelsList, outputLabelsList = readFiles("hw12data/digitsDataset/valLabels", "valOutput")