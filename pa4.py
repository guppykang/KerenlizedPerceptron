import collections
import math
import numpy
from random import randint


def loadData(fileName, features=[], labels=[]):
    fileOpen = open(fileName, "r")
    lines = fileOpen.readlines()
    fileOpen.close()
    for unFormattedLine in lines:
        line = unFormattedLine.split()
        features.append(line[0])

        if str(line[1][0]) == str('+'): 
            labels.append(1)
        else : 
            labels.append(-1)


def kernelFunction(first, second, p):
    count = 0 
    for start in range(0, len(first) - p):
        v = first[start : p - 1]
        count += second.count(v)
        
    return count


def predict(testX, misClassified, p): 
    sum = 0
    for wrong in misClassified: 
        sum += wrong[1] * kernelFunction(str(testX), str(wrong[0]), p)
    return numpy.sign(sum)


def kernenlizedPerceptron(trainingSet, trainingLabels, p):
    w = []
    for i in range(len(trainingSet)):
        print('hi mom')
        print(w)
        if int(trainingLabels[i]) * predict(trainingSet[i], w, p) <= 0: 
            w.append([i, trainingLabels[i]])

    return w


def getAccuracy(w, testingSet, testingLabels, p):
    numCorrect = 0
    for i in range(len(testingSet)):
        prediction = predict(testingSet[i], w, p) 
        if int(prediction) == testingLabels[i]:
            numCorrect += 1
        if int(prediction) == 0: 
            numCorrect += randint(0,1)
    
    return float(1) - float(numCorrect)/float(len(testingSet))




#main
trainFeatures = []
trainLabels = []

loadData('pa4train.txt', trainFeatures, trainLabels)

#train on substrings of size 3
classifier = kernenlizedPerceptron(trainFeatures, trainLabels, 5)
accuracy = getAccuracy(classifier, trainFeatures, trainLabels, 5)



