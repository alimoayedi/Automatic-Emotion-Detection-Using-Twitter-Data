import sys
sys.path.append(r"D:\Synced Folder\PyCharm Codes")
# import os

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import ModelDevelop
import ClassifierSelection


import csv
import copy
from matplotlib import pyplot as plot
import matplotlib.patches as mpatches
import random
import numpy as np
from sklearn import preprocessing
from sklearn import metrics


if __name__ == '__main__':

    rawScores = []
    rawScores_test = []
    trainLabels = []
    testLabels = []
    modelDevelop = ModelDevelop.ModelDevelop()
    clsSelection = ClassifierSelection.ClassifierSelection()
    
    tweetlength = 33

    with open(r"C:\Users\Ali\Desktop\anger-test data-train scores.csv") as trainFile:
        reader = csv.reader(trainFile)
        for value in reader:
            value = [ float(x) for x in value ]
            rawScores.append(value)
            
        featureScores_train = dict()
        
        selfDictLastOffset = 8 + (4*tweetlength)
    
        featureScores_train["featureOne"] = []
        featureScores_train["featureTwo"] = []
        featureScores_train["featureThr"] = []
        featureScores_train["featureFor"] = []
        featureScores_train["featureFiv"] = []
        featureScores_train["featureSix"] = []
        featureScores_train["featureSev"] = []
        featureScores_train["featureEit"] = []
        featureScores_train["featureNin"] = []
        featureScores_train["featureTen"] = []
        featureScores_train["featureElv"] = []
        featureScores_train["featureTlv"] = []
        featureScores_train["featureTrt"] = []
        featureScores_train["featureFrt"] = []
        featureScores_train["selfDict"] = []
        featureScores_train["query"] = []
        featureScores_train["symbol"] = []
        featureScores_train["tfidf"] = []
        featureScores_train["cosine"] = []
    
        for tweet in rawScores:
            featureScores_train["featureOne"].extend([tweet[0+(0* tweetlength ): tweetlength +(0* tweetlength )]])
            featureScores_train["featureTwo"].extend([tweet[0+(1* tweetlength ): tweetlength +(1* tweetlength )]])
            featureScores_train["featureThr"].extend([tweet[0+(2* tweetlength ): tweetlength +(2* tweetlength )]])
            featureScores_train["featureFor"].extend([tweet[0+(3* tweetlength ): tweetlength +(3* tweetlength )]])
            featureScores_train["featureFiv"].extend([tweet[0+(4* tweetlength ): tweetlength +(4* tweetlength )]])
            featureScores_train["featureSix"].extend([tweet[0+(5* tweetlength ): tweetlength +(5* tweetlength )]])
            featureScores_train["featureSev"].extend([tweet[0+(6* tweetlength ): tweetlength +(6* tweetlength )]])
            featureScores_train["featureEit"].extend([tweet[0+(7* tweetlength ): tweetlength +(7* tweetlength )]])
            featureScores_train["featureNin"].extend([tweet[0+(8* tweetlength ): tweetlength +(8* tweetlength )]])
            featureScores_train["featureTen"].extend([tweet[0+(9* tweetlength ): tweetlength +(9* tweetlength )]])
            featureScores_train["featureElv"].extend([tweet[0+(10* tweetlength ): tweetlength +(10* tweetlength )]])
            featureScores_train["featureTlv"].extend([tweet[0+(11* tweetlength ): tweetlength +(11* tweetlength )]])
            featureScores_train["featureTrt"].extend([tweet[0+(12* tweetlength ): tweetlength +(12* tweetlength )]])
            featureScores_train["featureFrt"].extend([tweet[0+(13* tweetlength ): tweetlength +(14* tweetlength )]])
            featureScores_train["tfidf"].extend([tweet[0+(15* tweetlength ): 4 +(15* tweetlength )]])
            featureScores_train["cosine"].extend([tweet[4+(15* tweetlength ):8+(15* tweetlength )]])
            featureScores_train["selfDict"].extend([tweet[8 + (15* tweetlength ): selfDictLastOffset + (15* tweetlength )]])
            featureScores_train["query"].extend([tweet[selfDictLastOffset + (15* tweetlength ): selfDictLastOffset + 4 + (15* tweetlength )]])
            featureScores_train["symbol"].extend([tweet[selfDictLastOffset + 4 + (15* tweetlength ): selfDictLastOffset + 8 + (15* tweetlength )]])
            trainLabels.append(str(int(tweet[-1])))
            
    with open(r"C:\Users\Ali\Desktop\anger-test data-test scores.csv") as testFile:
        reader = csv.reader(testFile)
        for value in reader:
            value = [ float(x) for x in value ]
            rawScores_test.append(value)
            
        featureScores_test = dict()
    
        featureScores_test["featureOne"] = []
        featureScores_test["featureTwo"] = []
        featureScores_test["featureThr"] = []
        featureScores_test["featureFor"] = []
        featureScores_test["featureFiv"] = []
        featureScores_test["featureSix"] = []
        featureScores_test["featureSev"] = []
        featureScores_test["featureEit"] = []
        featureScores_test["featureNin"] = []
        featureScores_test["featureTen"] = []
        featureScores_test["featureElv"] = []
        featureScores_test["featureTlv"] = []
        featureScores_test["featureTrt"] = []
        featureScores_test["featureFrt"] = []
        featureScores_test["selfDict"] = []
        featureScores_test["query"] = []
        featureScores_test["symbol"] = []
        featureScores_test["tfidf"] = []
        featureScores_test["cosine"] = []
    
        for tweet in rawScores_test:
            featureScores_test["featureOne"].extend([tweet[0+(0* tweetlength ): tweetlength +(0* tweetlength )]])
            featureScores_test["featureTwo"].extend([tweet[0+(1* tweetlength ): tweetlength +(1* tweetlength )]])
            featureScores_test["featureThr"].extend([tweet[0+(2* tweetlength ): tweetlength +(2* tweetlength )]])
            featureScores_test["featureFor"].extend([tweet[0+(3* tweetlength ): tweetlength +(3* tweetlength )]])
            featureScores_test["featureFiv"].extend([tweet[0+(4* tweetlength ): tweetlength +(4* tweetlength )]])
            featureScores_test["featureSix"].extend([tweet[0+(5* tweetlength ): tweetlength +(5* tweetlength )]])
            featureScores_test["featureSev"].extend([tweet[0+(6* tweetlength ): tweetlength +(6* tweetlength )]])
            featureScores_test["featureEit"].extend([tweet[0+(7* tweetlength ): tweetlength +(7* tweetlength )]])
            featureScores_test["featureNin"].extend([tweet[0+(8* tweetlength ): tweetlength +(8* tweetlength )]])
            featureScores_test["featureTen"].extend([tweet[0+(9* tweetlength ): tweetlength +(9* tweetlength )]])
            featureScores_test["featureElv"].extend([tweet[0+(10* tweetlength ): tweetlength +(10* tweetlength )]])
            featureScores_test["featureTlv"].extend([tweet[0+(11* tweetlength ): tweetlength +(11* tweetlength )]])
            featureScores_test["featureTrt"].extend([tweet[0+(12* tweetlength ): tweetlength +(12* tweetlength )]])
            featureScores_test["featureFrt"].extend([tweet[0+(13* tweetlength ): tweetlength +(14* tweetlength )]])
            featureScores_test["tfidf"].extend([tweet[0+(15* tweetlength ): 4 +(15* tweetlength )]])
            featureScores_test["cosine"].extend([tweet[4+(15* tweetlength ):8+(15* tweetlength )]])
            featureScores_test["selfDict"].extend([tweet[8 + (15* tweetlength ): selfDictLastOffset + (15* tweetlength )]])
            featureScores_test["query"].extend([tweet[selfDictLastOffset + (15* tweetlength ): selfDictLastOffset + 4 + (15* tweetlength )]])
            featureScores_test["symbol"].extend([tweet[selfDictLastOffset + 4 + (15* tweetlength ): selfDictLastOffset + 8 + (15* tweetlength )]])
            testLabels.append(str(int(tweet[-1])))
            
        ###############################################################################
        # Randomized Forward Selection ################################################
        ###############################################################################
    
#    scaler = preprocessing.Normalizer()
#    performances = dict()
#    scoreSet_train = []
#    scoreSet_test = []
#    listOfFeatures = [feature for feature in featureScores_train.keys()]
#    listOfBests = []
#    listOfCandidateFeatures = []
#    listOfSelectedFeatures = []
#    bestPerformance = 0
#    bestCurrent = 0
#    noOfFeatures = len(listOfFeatures)
#    repeat = True
#    for feature in listOfFeatures:
#        scoreSet_train.clear()
#        scoreSet_test.clear()
#        scalingModel = scaler.fit(featureScores_train[feature])
#        scaledTrain = scalingModel.transform(featureScores_train[feature])
#        scaledTest = scalingModel.transform(featureScores_test[feature])
#        performances[feature] = modelDevelop.trainModel(scaledTrain, trainLabels, scaledTest, testLabels)['f_micro'].tolist()
#        if performances[feature] >= bestCurrent:
#            if performances[feature] == bestCurrent:
#                listOfBests.append(feature)
#            else:
#                listOfBests.clear()
#                listOfBests.append(feature)
#                bestCurrent = performances[feature]
#    selectedFeature = listOfBests[random.randint(0,len(listOfBests)-1)]
#    listOfFeatures.remove(selectedFeature)
#    while repeat:
#        candidateFeature = listOfFeatures[random.randint(0,len(listOfFeatures)-1)]
#        listOfCandidateFeatures.append(candidateFeature)
#        listOfFeatures.remove(candidateFeature)
#        key = selectedFeature
#        scoreSet_train.clear()
#        scoreSet_test.clear()
#        for tweet in featureScores_train[selectedFeature]:
#            scoreSet_train.append(tweet.copy())
#        for tweet in featureScores_test[selectedFeature]:
#            scoreSet_test.append(tweet.copy())
#        for survivedFeature in listOfCandidateFeatures:
#            key = key + " " + survivedFeature
#            for index, tweet in enumerate(featureScores_train[survivedFeature]):
#                scoreSet_train[index].extend(tweet.copy())
#            for index, tweet in enumerate(featureScores_test[survivedFeature]):
#                scoreSet_test[index].extend(tweet.copy())
#        performances[key] = modelDevelop.trainModel(scoreSet_train, trainLabels, scoreSet_test, testLabels)['f_micro'].tolist()
#        print(len(scoreSet_train[0]))
#        print(key + "\t" + str(performances[key]))
#        if performances[key] >= bestCurrent:
#            listOfSelectedFeatures = listOfCandidateFeatures.copy()
#            bestCurrent = performances[key]
#        elif bestCurrent > bestPerformance or len(listOfSelectedFeatures) == noOfFeatures:
#            listOfSelectedFeatures.insert(0,selectedFeature)
#            repeat = False

    ###############################################################################
    # Forward Selection ###########################################################
    ###############################################################################

    scaler = preprocessing.Normalizer()
    performances = dict()
    scoreSet_train = []
    scoreSet_test = []
    listOfFeatures = [feature for feature in featureScores_train.keys()]
    listOfSelectedFeatures = []
    bestPerformance = 0
    bestCurrent = 0
    noOfFeatures = len(listOfFeatures)
    repeat = True
    while repeat:
        candidateFeatures = []
        for feature in listOfFeatures:
            key = feature
            scoreSet_train.clear()
            scoreSet_test.clear()
            for tweet in featureScores_train[feature]:
                scoreSet_train.append(tweet.copy())
            for tweet in featureScores_test[feature]:
                scoreSet_test.append(tweet.copy())
            for survivedFeature in listOfSelectedFeatures:
                key = key + " " + survivedFeature
                for index, tweet in enumerate(featureScores_train[survivedFeature]):
                    scoreSet_train[index].extend(tweet.copy())
                for index, tweet in enumerate(featureScores_test[survivedFeature]):
                    scoreSet_test[index].extend(tweet.copy())
            scalingModel = scaler.fit(scoreSet_train)
            scaledTrain = scalingModel.transform(scoreSet_train)
            scaledTest = scalingModel.transform(scoreSet_test)
            performances[key] = modelDevelop.trainModel(scaledTrain, trainLabels, scaledTest, testLabels)['f_micro'].tolist()
            print(key + "\t" + str(performances[key]))
            if performances[key] > bestCurrent:
                bestCurrent = performances[key]
                candidateFeatures.clear()
                candidateFeatures.append(feature)
            elif performances[key] == bestCurrent:
                candidateFeatures.append(feature)
        if bestCurrent < bestPerformance or len(listOfSelectedFeatures) == noOfFeatures or len(candidateFeatures) == 0:
            repeat = False
        else:
            electedFeature = candidateFeatures[int(np.random.choice(len(candidateFeatures), 1))]
            listOfSelectedFeatures.extend([electedFeature])
            bestPerformance = bestCurrent
            listOfFeatures.remove(electedFeature)
    
    ###############################################################################
    # Simplified Forward Selection ################################################
    ###############################################################################
   
#    scaler = preprocessing.Normalizer()
#    performances = dict()
#    scoreSet_train = []
#    scoreSet_test = []
#    listOfFeatures = [feature for feature in featureScores_train.keys()]
#    listOfSelectedFeatures = []
#    bestPerformance = 0
#    bestCurrent = 0
#    noOfFeatures = len(listOfFeatures)
#    repeat = True
#    for feature in listOfFeatures:
#        key = feature
#        scoreSet_train.clear()
#        scoreSet_test.clear()
#        for tweet in featureScores_train[feature]:
#            scoreSet_train.append(tweet.copy())
#        for tweet in featureScores_test[feature]:
#            scoreSet_test.append(tweet.copy())        
#        scalingModel = scaler.fit(scoreSet_train)
#        scaledTrain = scalingModel.transform(scoreSet_train)
#        scaledTest = scalingModel.transform(scoreSet_test)
#        performances[key] = modelDevelop.trainModel(scaledTrain, trainLabels, scaledTest, testLabels)['f_micro'].tolist()
#    while repeat:
#        if len(listOfSelectedFeatures) == 0:
#            candidateList = performances.copy()
#            listOfBests = [key for key in candidateList.keys() if candidateList[key] == max(list(candidateList.values()))]
#            selectedFeature = listOfBests[random.randint(0,len(listOfBests)-1)]
#            listOfSelectedFeatures.extend([selectedFeature])
#            bestPerformance = performances[selectedFeature]
#            candidateList.pop(selectedFeature)
#        listOfBests = [key for key in candidateList.keys() if candidateList[key] == max(list(candidateList.values()))]
#        selectedFeature = listOfBests[random.randint(0,len(listOfBests)-1)]
#        key = selectedFeature
#        scoreSet_train.clear()
#        scoreSet_test.clear()
#        for tweet in featureScores_train[selectedFeature]:
#            scoreSet_train.append(tweet.copy())
#        for tweet in featureScores_test[selectedFeature]:
#            scoreSet_test.append(tweet.copy())
#        for survivedFeature in listOfSelectedFeatures:
#            key = key + " " + survivedFeature
#            for index, tweet in enumerate(featureScores_train[survivedFeature]):
#                scoreSet_train[index].extend(tweet.copy())
#            for index, tweet in enumerate(featureScores_test[survivedFeature]):
#                scoreSet_test[index].extend(tweet.copy())
#        scalingModel = scaler.fit(scoreSet_train)
#        scaledTrain = scalingModel.transform(scoreSet_train)
#        scaledTest = scalingModel.transform(scoreSet_test)
#        performances[key] = modelDevelop.trainModel(scaledTrain, trainLabels, scaledTest, testLabels)['f_micro'].tolist()
#        print(key + "\t" + str(performances[key]))
#        bestCurrent = performances[key]
#        if bestPerformance <= bestCurrent:
#            bestPerformance = bestCurrent
#            listOfSelectedFeatures.extend([selectedFeature])
#            candidateList.pop(selectedFeature)
#        else:
#            repeat = False
#    print("best feature set is ")
#    for feature in listOfSelectedFeatures:
#        print(feature)
    
    ###############################################################################
    # Backward Selection ##########################################################
    ###############################################################################

#    scaler = preprocessing.Normalizer()
#    performances = dict()
#    scoreSet_train = []
#    scoreSet_test = []
#    listOfFeatures = [feature for feature in featureScores_train.keys()]
#    iterator = 0
#    bestPerformance = 0
#    bestCurrent = 0
#    droppedFeatures = []
#    noOfFeatures = len(listOfFeatures)
#    repeat = True
#    while repeat:
#        scoreSet_train.clear()
#        scoreSet_test.clear()
#        scoreSet_train = [[] for i in range(0, len(featureScores_train[listOfFeatures[0]]))]
#        scoreSet_test = [[] for i in range(0, len(featureScores_test[listOfFeatures[0]]))]
#        for feature in listOfFeatures:
#            for index, tweet in enumerate(featureScores_train[feature]):
#                scoreSet_train[index].extend(tweet.copy())
#            for index, tweet in enumerate(featureScores_test[feature]):
#                scoreSet_test[index].extend(tweet.copy())
#        scalingModel = scaler.fit(scoreSet_train)
#        scaledTrain = scalingModel.transform(scoreSet_train)
#        scaledTest = scalingModel.transform(scoreSet_test)
#        performances[iterator] = modelDevelop.trainModel(scaledTrain, trainLabels, scaledTest, testLabels)['f_micro'].tolist()
#        print(str(iterator) + "\t" + str(performances[iterator]))
#        if performances[iterator] >= bestPerformance:
#            dropFeatureIndex = int(np.random.choice(len(listOfFeatures), 1))
#            candidateFeature = [dropFeatureIndex, listOfFeatures[dropFeatureIndex]]
#            droppedFeatures.append(candidateFeature)
#            del listOfFeatures[dropFeatureIndex]
#            bestPerformance = performances[iterator]
#            iterator = iterator + 1
#        else:
#            listOfFeatures.insert(candidateFeature[0], candidateFeature[1])
#            droppedFeatures.remove(candidateFeature)
#            repeat = False
#    print("best feature set is ")
#    for feature in listOfFeatures:
#        print(feature)
    
    ###############################################################################
    # Classifier Selection ########################################################
    ###############################################################################
    
    scaler = preprocessing.Normalizer()
    performances = dict()
    predicts = dict()
    classifierPerformances = dict()
    listOfFeatures = [feature for feature in featureScores_train.keys()]
    listOfBests = []
    listOfCandidateFeatures = []
    listOfSelectedFeatures = []
    bestPerformance = 0
    bestCurrent = 0
    noOfFeatures = len(listOfFeatures)
    repeat = True
    for feature in listOfFeatures:
        scalingModel = scaler.fit(featureScores_train[feature])
        scaledTrain = scalingModel.transform(featureScores_train[feature])
        scaledTest = scalingModel.transform(featureScores_test[feature])
        performances[feature] = modelDevelop.trainModel(scaledTrain, trainLabels, scaledTest, testLabels)
        predicts[feature] = modelDevelop.getPredictedLabels()
        
    microFscoreWeights = dict()
    levelWiseFscoreWeights = dict()
    macroFscoreWeights = dict()
    for key, value in performances.items():
        microFscoreWeights[key] = performances[key]['f_micro']
        levelWiseFscoreWeights[key] = performances[key]['f_level']
        macroFscoreWeights[key] =  sum(levelWiseFscoreWeights[key])/4
    
    stdMicroFscoreWeights = copy.deepcopy(microFscoreWeights)
    stdMacroFscoreWeights = copy.deepcopy(macroFscoreWeights)
    stdLevelWiseFscoreWeights = copy.deepcopy(levelWiseFscoreWeights)
    for key in stdMicroFscoreWeights.keys():
        stdMicroFscoreWeights[key] = stdMicroFscoreWeights[key]/sum(microFscoreWeights.values())
        stdMacroFscoreWeights[key] = stdMacroFscoreWeights[key]/sum(macroFscoreWeights.values())
        stdLevelWiseFscoreWeights[key]=(np.array(levelWiseFscoreWeights[key])/sum(levelWiseFscoreWeights[key])).tolist()
    
    weightDict = macroFscoreWeights
    weightFormat = 'classifierwise'
        
    # Baseline Selection ##########################################################

#    clsSelection.tweetVoting(predicts, None)
#    classifiedPredictions = clsSelection.getPredictedLabels()
#    metrics.precision_recall_fscore_support(testLabels, classifiedPredictions)
#    metrics.f1_score(testLabels, classifiedPredictions, average='micro')
#    metrics.precision_recall_fscore_support(testLabels, classifiedPredictions,average='macro')
    
    # Random Forward Selection #####################################################

    listOfFeatures.pop(2)
    
    selectedPredicts = dict()
    selectedWeights = dict()
    for feature in listOfFeatures:
        selectedPredicts.clear()
        selectedWeights.clear()
        selectedPredicts[feature] = predicts[feature]
        selectedWeights[feature] = weightDict[feature]
        clsSelection.tweetVoting(selectedPredicts, selectedWeights, weightFormat)
        classifiedPredictions = clsSelection.getPredictedLabels()
        classifierPerformances[feature] = metrics.f1_score(testLabels, classifiedPredictions, average='micro')
        if classifierPerformances[feature] >=  bestPerformance:
            if classifierPerformances[feature] > bestPerformance:
                listOfCandidateFeatures.clear()    
            bestPerformance = classifierPerformances[feature]
            listOfCandidateFeatures.append(feature)
    selectedFeature = listOfCandidateFeatures[int(np.random.choice(len(listOfCandidateFeatures), 1))]
    listOfSelectedFeatures.append(selectedFeature)
    listOfFeatures.remove(selectedFeature)
    while(repeat):
        selectedPredicts.clear()
        selectedWeights.clear()
        key = ""
        for feature in listOfSelectedFeatures:
            key = key + feature
            selectedPredicts[feature] = predicts[feature]
            selectedWeights[feature] = weightDict[feature]
        electedFeature = listOfFeatures[int(np.random.choice(len(listOfFeatures), 1))]
        key = key + " " + electedFeature
        listOfFeatures.remove(electedFeature)
        listOfSelectedFeatures.append(electedFeature)
        selectedPredicts[electedFeature] = predicts[electedFeature]
        selectedWeights[electedFeature] = weightDict[electedFeature]
        
        clsSelection.tweetVoting(selectedPredicts, selectedWeights, weightFormat)
        classifiedPredictions = clsSelection.getPredictedLabels()
        classifierPerformances[key] = metrics.f1_score(testLabels, classifiedPredictions, average='micro')
        
        if len(listOfFeatures) == 0 or classifierPerformances[key] < bestPerformance:
            repeat = False
        else:
            bestPerformance = classifierPerformances[key]
        print(classifierPerformances)
    
    # Backward Selection ###########################################################
    listOfFeatures.pop(1)

    selectedPredicts = dict()
    selectedWeights = dict()
    
    selectedPredicts = predicts.copy()
    selectedWeights = weightDict.copy()
    
    key = "All"

    while(repeat):
        clsSelection.tweetVoting(selectedPredicts, selectedWeights, weightFormat)
        classifiedPredictions = clsSelection.getPredictedLabels()
        classifierPerformances[key] = metrics.f1_score(testLabels, classifiedPredictions, average='micro')
        print(key + str(classifierPerformances[key]))
        if classifierPerformances[key] < bestPerformance or len(listOfFeatures) == 1:
            repeat=False
        else:
            bestPerformance = classifierPerformances[key]
            electedFeature = listOfFeatures[int(np.random.choice(len(listOfFeatures), 1))]
            listOfFeatures.remove(electedFeature)
            del selectedPredicts[electedFeature]
            del selectedWeights[electedFeature]
            key = electedFeature
    
    # Simplified Forward Selection #################################################
    
    listOfFeatures.pop(2)
    
    selectedPredicts = dict()
    selectedWeights = dict()
    
    for feature in listOfFeatures:
        selectedPredicts.clear()
        selectedWeights.clear()
        selectedPredicts[feature] = predicts[feature]
        selectedWeights[feature] = weightDict[feature]
        clsSelection.tweetVoting(selectedPredicts, selectedWeights, weightFormat)
        classifiedPredictions = clsSelection.getPredictedLabels()
        classifierPerformances[feature] = metrics.f1_score(testLabels, classifiedPredictions, average='micro')
        if classifierPerformances[feature] >=  bestPerformance:
            if classifierPerformances[feature] > bestPerformance:
                listOfBests.clear()    
            bestPerformance = classifierPerformances[feature]
            listOfBests.append(feature)
    candidateFeatures = classifierPerformances.copy()
    selectedFeature = listOfBests[int(np.random.choice(len(listOfBests), 1))]
    listOfSelectedFeatures.append(selectedFeature)
    del candidateFeatures[selectedFeature]
    while(repeat):
        selectedPredicts.clear()
        selectedWeights.clear()
        key = ""
        electedFeature = max(candidateFeatures, key=candidateFeatures.get)
        listOfSelectedFeatures.append(electedFeature)
        for feature in listOfSelectedFeatures:
            key = key + " " + feature
            selectedPredicts[feature] = predicts[feature]
            selectedWeights[feature] = weightDict[feature]
        
        clsSelection.tweetVoting(selectedPredicts, selectedWeights, weightFormat)
        classifiedPredictions = clsSelection.getPredictedLabels()
        classifierPerformances[key] = metrics.f1_score(testLabels, classifiedPredictions, average='micro')
        
        if len(listOfFeatures) == 0:# or classifierPerformances[key] < bestPerformance:
            repeat = False
        else:
            bestPerformance = classifierPerformances[key]
            del candidateFeatures[electedFeature]
        print(classifierPerformances)

    # Forward Selection #################################################
    
    listOfFeatures.pop(3)
    
    selectedPredicts = dict()
    selectedWeights = dict()
    
    for feature in listOfFeatures:
        selectedPredicts.clear()
        selectedWeights.clear()
        selectedPredicts[feature] = predicts[feature]
        selectedWeights[feature] = weightDict[feature]
        clsSelection.tweetVoting(selectedPredicts, selectedWeights, weightFormat)
        classifiedPredictions = clsSelection.getPredictedLabels()
        classifierPerformances[feature] = metrics.f1_score(testLabels, classifiedPredictions, average='micro')
        print(feature + " " + str(classifierPerformances[feature]))
        if classifierPerformances[feature] >=  bestPerformance:
            if classifierPerformances[feature] > bestPerformance:
                listOfBests.clear()    
            bestPerformance = classifierPerformances[feature]
            listOfBests.append(feature)
    selectedFeature = listOfBests[int(np.random.choice(len(listOfBests), 1))]
    listOfSelectedFeatures.append(selectedFeature)
    listOfFeatures.remove(selectedFeature)
    while(repeat):
        for candidateFeature in listOfFeatures:
            selectedPredicts.clear()
            selectedWeights.clear()
            selectedPredicts[feature] = predicts[feature]
            selectedWeights[feature] = weightDict[feature]
            key = candidateFeature
            for feature in listOfSelectedFeatures:
                key = key + " " + feature
                selectedPredicts[feature] = predicts[feature]
                selectedWeights[feature] = weightDict[feature]
            clsSelection.tweetVoting(selectedPredicts, selectedWeights, weightFormat)
            classifiedPredictions = clsSelection.getPredictedLabels()
            classifierPerformances[key] = metrics.f1_score(testLabels, classifiedPredictions, average='micro')
            print(key + " " + str(classifierPerformances[key]))
            if classifierPerformances[key] >= bestCurrent:
                if classifierPerformances[key] > bestCurrent:
                    listOfCandidateFeatures.clear()
                listOfCandidateFeatures.append(candidateFeature)
                bestCurrent = classifierPerformances[key]
        if bestCurrent < bestPerformance or len(listOfFeatures)==0 or len(listOfCandidateFeatures)==0:
            repeat = False
        else:
            electedFeature=listOfCandidateFeatures[int(np.random.choice(len(listOfCandidateFeatures), 1))]
            listOfFeatures.remove(electedFeature)
            listOfSelectedFeatures.append(electedFeature)
            bestPerformance = bestCurrent
            listOfCandidateFeatures.clear()

    
    
    
    
    
    
    
    
    
    