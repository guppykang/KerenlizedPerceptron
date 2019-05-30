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
    substringsInSecond = []
    for start in range(0, len(second) - p + 1):
        v = second[start : start + p]
        substringsInSecond.append(v)
        
    for start in range(0, len(first) - p + 1):
        v = first[start : start + p]
        count += substringsInSecond.count(v) 
    
    return count

def uniqueKernelFunction(first, second, p):

    count = 0 
    substringsInSecond = []
    for start in range(0, len(second) - p + 1):
        v = second[start : start + p]
        if v not in substringsInSecond: 
            substringsInSecond.append(v)
        
    #print('substrings in second : ' + str(substringsInSecond))

    uniqueSubstringsInFirst = []
    for start in range(0, len(first) - p + 1):
        v = first[start : start + p]
        if v not in uniqueSubstringsInFirst: 
            # print('in here')

            #print('count for  ' + v  + ' ' + str(substringsInSecond.count(v)))
            count += substringsInSecond.count(v) 
            uniqueSubstringsInFirst.append(v)
    
    #print('total count : '+ str(count))
    return count

def predict(testX, misClassified, p, isUnique): 
    sum = 0
    
    for wrong in misClassified: 
        if isUnique: 
            sum += wrong[1] * uniqueKernelFunction(str(testX), str(wrong[0]), p)
        elif not isUnique: 
            sum += wrong[1] * kernelFunction(str(testX), str(wrong[0]), p)
    return numpy.sign(sum)


def kernenlizedPerceptron(trainingSet, trainingLabels, p, isUnique):
    w = []

    #print(trainingSet)
    #print(trainingLabels)

    for i in range(len(trainingSet)):
        # print('')
        # print('ROUND : ' + str(i) + ' OF ' + str(len(trainingSet)))
        # for item in w: 
        #     print(trainingSet.index(item[0]))

        prediction = predict(trainingSet[i], w, p, isUnique)
        #print('prediction : ' + str(prediction) + '. acutal : ' + str(trainingLabels[i]))
        if int(trainingLabels[i]) * prediction <= 0: 
            #print('incorrect')
            w.append([trainingSet[i], trainingLabels[i]])
        #print('hi mom')

    return w


def getAccuracy(w, testingSet, testingLabels, p, isUnique):
    numCorrect = 0
    for i in range(len(testingSet)):
        # print('')
        # print('Testing ROUND : ' + str(i) + ' OF ' + str(len(testingSet)))

        prediction = predict(testingSet[i], w, p, isUnique) 
        if int(prediction) == testingLabels[i]:
            numCorrect += 1
        if int(prediction) == 0: 
            numCorrect += randint(0,1)
    
    return float(1) - float(numCorrect)/float(len(testingSet))




#main
trainFeatures = []
trainLabels = []
testingFeatures = []
testingLabels = []

loadData('pa4train.txt', trainFeatures, trainLabels)
loadData('pa4test.txt', testingFeatures, testingLabels)
#loadData('testing.txt', trainFeatures, trainLabels)


# print("p = 2: ")
# classifier = kernenlizedPerceptron(trainFeatures, trainLabels, 2, True)
# accuracy = getAccuracy(classifier, trainFeatures, trainLabels, 2, True)
# print(accuracy)

print("p = 3: ")
classifier = kernenlizedPerceptron(trainFeatures, trainLabels, 3, True)
accuracy = getAccuracy(classifier, trainFeatures, trainLabels, 3, True)
print('training : ' + str(accuracy))
accuracy = getAccuracy(classifier, testingFeatures, testingLabels, 3, True)
print('testing : ' + str(accuracy))


print("p = 4: ")
classifier = kernenlizedPerceptron(trainFeatures, trainLabels, 4, True)
accuracy = getAccuracy(classifier, trainFeatures, trainLabels, 4, True)
print('training : ' + str(accuracy))
accuracy = getAccuracy(classifier, testingFeatures, testingLabels, 4, True)
print('testing : ' + str(accuracy))


print("p = 5: ")
classifier = kernenlizedPerceptron(trainFeatures, trainLabels, 5, True)
accuracy = getAccuracy(classifier, trainFeatures, trainLabels, 5, True)
print('training : ' + str(accuracy))
accuracy = getAccuracy(classifier, testingFeatures, testingLabels, 5, True)
print('testing : ' + str(accuracy))

print("p = 3: ")
classifier = kernenlizedPerceptron(trainFeatures, trainLabels, 3, False)
accuracy = getAccuracy(classifier, trainFeatures, trainLabels, 3, False)
print('training : ' + str(accuracy))
accuracy = getAccuracy(classifier, testingFeatures, testingLabels, 3, False)
print('testing : ' + str(accuracy))

print("p = 4: ")
classifier = kernenlizedPerceptron(trainFeatures, trainLabels, 4, False)
accuracy = getAccuracy(classifier, trainFeatures, trainLabels, 4, False)
print('training : ' + str(accuracy))
accuracy = getAccuracy(classifier, testingFeatures, testingLabels, 4, False)
print('testing : ' + str(accuracy))

print("p = 5: ")
classifier = kernenlizedPerceptron(trainFeatures, trainLabels, 5, False)
accuracy = getAccuracy(classifier, trainFeatures, trainLabels, 5, False)
print('training : ' + str(accuracy))
accuracy = getAccuracy(classifier, testingFeatures, testingLabels, 5, False)
print('testing : ' + str(accuracy))

