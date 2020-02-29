import sys
sys.path.append(r"D:\Synced Folder\PyCharm Codes")
# import os

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import TestTrainFiles
import Lexicons
import TweetGeneralScoring
import ModelDevelop

import random
import numpy as np
from sklearn import preprocessing

if __name__ == '__main__':
    testTrainFiles = TestTrainFiles.TestTrainFiles()
    lexicons = Lexicons.Lexicons()
    modelDevelop = ModelDevelop.ModelDevelop()

    trainFile = testTrainFiles.getFile('train')[0]
    developFile = testTrainFiles.getFile('develop')[0]

    tweetScoring = TweetGeneralScoring.TweetGeneralScoring()

    tweetScoring.fit(trainFile=trainFile)
    tweetScoring.fitTest(testFile=developFile)
    trainLabels = tweetScoring.getLabels('train')
    testLabels = tweetScoring.getLabels('test')

    # set tweet size based on emotion
    if trainFile[1] == 'anger':
        normalize = True  # checked
        tweetSize = 34  # selected by checking micro f-scores for all lexicons. (34)
        topTermsPerc = 30  # selected by checking micro f-scores not individual classes.
        lexNineScoreAdd = r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\lexNineScores\anger_lexNineScore_train.csv"
        lexNineScoreAdd_test = r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\lexNineScores\anger_lexNineScore_test.csv"
    elif trainFile[1] == 'joy':
        normalize = True  # checked, actually they don't differ
        tweetSize = 31  # selected by checking micro f-scores for all lexicons. (31)
        topTermsPerc = 75  # selected by checking micro f-scores not individual classes.
        lexNineScoreAdd = r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\lexNineScores\joy_lexNineScore_train.csv"
        lexNineScoreAdd_test = r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\lexNineScores\joy_lexNineScore_test.csv"
    elif trainFile[1] == 'fear':
        normalize = True  # checked, no improve in micro f_scores
        tweetSize = 36  # selected by checking micro f-scores for all lexicons. (18)
        topTermsPerc = 15  # different percentages work equal
        lexNineScoreAdd = r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\lexNineScores\fear_lexNineScore_train.csv"
        lexNineScoreAdd_test = r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\lexNineScores\fear_lexNineScore_test.csv"
    else:
        normalize = True  # checked, individual performance increase however micro f_Score of all features decreases
        tweetSize = 29  # selected by checking micro f-scores for all lexicons. (30)
        topTermsPerc = 75  # selected by checking micro f-scores not individual classes.
        lexNineScoreAdd = r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\lexNineScores\sadness_lexNineScore_train.csv"
        lexNineScoreAdd_test = r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\lexNineScores\sadness_lexNineScore_test.csv"

    featureScores_train = dict()
    featureScores_test = dict()

    featureScores_train["featureOne"] = tweetScoring.lexiconScoring('train', tweetSize, lexicons.getLexicons(trainFile[1])[0])
    featureScores_train["featureTwo"] = tweetScoring.lexiconScoring('train', tweetSize, lexicons.getLexicons(trainFile[1])[1])
    featureScores_train["featureThr"] = tweetScoring.lexiconScoring('train', tweetSize, lexicons.getLexicons(trainFile[1])[2])
    featureScores_train["featureFor"] = tweetScoring.lexiconScoring('train', tweetSize, lexicons.getLexicons(trainFile[1])[3])
    featureScores_train["featureFiv"] = tweetScoring.lexiconScoring('train', tweetSize, lexicons.getLexicons(trainFile[1])[4])
    featureScores_train["featureSix"] = tweetScoring.lexiconScoring('train', tweetSize, lexicons.getLexicons(trainFile[1])[5])
    featureScores_train["featureSev"] = tweetScoring.lexiconScoring('train', tweetSize, lexicons.getLexicons(trainFile[1])[6])
    featureScores_train["featureEit"] = tweetScoring.lexiconScoring('train', tweetSize, lexicons.getLexicons(trainFile[1])[7])
    featureScores_train["featureNin"] = tweetScoring.lexiconNineScoring('train', lexNineScoreAdd, tweetSize)
    featureScores_train["featureTen"] = tweetScoring.lexiconScoring('train', tweetSize, lexicons.getLexicons(trainFile[1])[8])
    featureScores_train["featureElv"] = tweetScoring.lexiconScoring('train', tweetSize, lexicons.getLexicons(trainFile[1])[9])
    featureScores_train["featureTlv"] = tweetScoring.lexiconScoring('train', tweetSize,
                                                                    lexicons.getLexicons(trainFile[1])[10])
    featureScores_train["featureTrt"] = tweetScoring.lexiconScoring('train', tweetSize,
                                                                    lexicons.getLexicons(trainFile[1])[11])
    featureScores_train["featureFrt"] = tweetScoring.lexiconScoring('train', tweetSize,
                                                                    lexicons.getLexicons(trainFile[1])[12])  # Bing Lui
    biwordBingLui = tweetScoring.lexiconScoring('train', tweetSize, lexicons.getLexicons(trainFile[1])[13])
    for index, tweetScore in enumerate(biwordBingLui):
        featureScores_train["featureFrt"][index][2].extend(tweetScore[2])
    featureScores_train["selfDict"] = tweetScoring.selfDictScoring('train', topTermsPerc, tweetSize)
    featureScores_train["query"] = tweetScoring.queryTermScoring('train', trainFile[1])
    featureScores_train["symbol"] = tweetScoring.symbolScoring('train')
    featureScores_train["tfidf"] = tweetScoring.tfidfScoring('train')
    featureScores_train["cosine"] = tweetScoring.cosineScoring('train')

    print("############# training done #############")

    featureScores_test["featureOne"] = tweetScoring.lexiconScoring('test', tweetSize, lexicons.getLexicons(developFile[1])[0])
    featureScores_test["featureTwo"] = tweetScoring.lexiconScoring('test', tweetSize, lexicons.getLexicons(developFile[1])[1])
    featureScores_test["featureThr"] = tweetScoring.lexiconScoring('test', tweetSize, lexicons.getLexicons(developFile[1])[2])
    featureScores_test["featureFor"] = tweetScoring.lexiconScoring('test', tweetSize, lexicons.getLexicons(developFile[1])[3])
    featureScores_test["featureFiv"] = tweetScoring.lexiconScoring('test', tweetSize, lexicons.getLexicons(developFile[1])[4])
    featureScores_test["featureSix"] = tweetScoring.lexiconScoring('test', tweetSize, lexicons.getLexicons(developFile[1])[5])
    featureScores_test["featureSev"] = tweetScoring.lexiconScoring('test', tweetSize, lexicons.getLexicons(developFile[1])[6])
    featureScores_test["featureEit"] = tweetScoring.lexiconScoring('test', tweetSize, lexicons.getLexicons(developFile[1])[7])
    featureScores_test["featureNin"] = tweetScoring.lexiconNineScoring('test', lexNineScoreAdd_test, tweetSize)
    featureScores_test["featureTen"] = tweetScoring.lexiconScoring('test', tweetSize, lexicons.getLexicons(developFile[1])[8])
    featureScores_test["featureElv"] = tweetScoring.lexiconScoring('test', tweetSize, lexicons.getLexicons(developFile[1])[9])
    featureScores_test["featureTlv"] = tweetScoring.lexiconScoring('test', tweetSize,
                                                                   lexicons.getLexicons(developFile[1])[10])
    featureScores_test["featureTrt"] = tweetScoring.lexiconScoring('test', tweetSize,
                                                                   lexicons.getLexicons(developFile[1])[11])
    featureScores_test["featureFrt"] = tweetScoring.lexiconScoring('test', tweetSize,
                                                                   lexicons.getLexicons(developFile[1])[12])  # Bing Lui
    biwordBingLui = tweetScoring.lexiconScoring('test', tweetSize, lexicons.getLexicons(developFile[1])[13])
    for index, tweetScore in enumerate(biwordBingLui):
        featureScores_test["featureFrt"][index][2].extend(tweetScore[2])
    featureScores_test["selfDict"] = tweetScoring.selfDictScoring('test', topTermsPerc, tweetSize)
    featureScores_test["query"] = tweetScoring.queryTermScoring('test', developFile[1])
    featureScores_test["symbol"] = tweetScoring.symbolScoring('test')
    featureScores_test["tfidf"] = tweetScoring.tfidfScoring('test')
    featureScores_test["cosine"] = tweetScoring.cosineScoring('test')

    print("############# testing done #############")

    ###############################################################################
    # Randomized Forward Selection ###########################################################
    ###############################################################################

    scaler = preprocessing.Normalizer()
    performances = dict()
    scoreSet_train = []
    scoreSet_test = []
    listOfFeatures = [feature for feature in featureScores_train.keys()]
    listOfBests = []
    listOfCandidateFeatures = []
    listOfSelectedFeatures = []
    bestPerformance = 0
    bestCurrent = 0
    noOfFeatures = len(listOfFeatures)
    repeat = True
    for feature in listOfFeatures:
        scoreSet_train.clear()
        scoreSet_test.clear()
        for tweet in featureScores_train[feature]:
            scoreSet_train.append(tweet[2])
        for tweet in featureScores_test[feature]:
            scoreSet_test.append(tweet[2])
        scalingModel = scaler.fit(scoreSet_train)
        scaledTrain = scalingModel.transform(scoreSet_train)
        scaledTest = scalingModel.transform(scoreSet_test)
        performances[feature] = modelDevelop.trainModel(scaledTrain, trainLabels, scaledTest, testLabels)['f_micro'].tolist()
        if performances[feature] >= bestCurrent:
            if performances[feature] == bestCurrent:
                listOfBests.append(feature)
            else:
                listOfBests.clear()
                listOfBests.append(feature)
                bestCurrent = performances[feature]
    selectedFeature = listOfBests[random.randint(0,len(listOfBests)-1)]
    listOfFeatures.remove(selectedFeature)
    while repeat:
        candidateFeature = listOfFeatures[random.randint(0,len(listOfFeatures)-1)]
        listOfCandidateFeatures.append(candidateFeature)
        listOfFeatures.remove(candidateFeature)
        key = selectedFeature
        scoreSet_train.clear()
        scoreSet_test.clear()
        for tweet in featureScores_train[selectedFeature]:
            scoreSet_train.append(tweet[2])
        for tweet in featureScores_test[selectedFeature]:
            scoreSet_test.append(tweet[2])
        for survivedFeature in listOfCandidateFeatures:
            key = key + " " + survivedFeature
            for index, tweet in enumerate(featureScores_train[survivedFeature]):
                scoreSet_train[index].extend(tweet[2])
            for index, tweet in enumerate(featureScores_test[survivedFeature]):
                scoreSet_test[index].extend(tweet[2])
        scalingModel = scaler.fit(scoreSet_train)
        scaledTrain = scalingModel.transform(scoreSet_train)
        scaledTest = scalingModel.transform(scoreSet_test)
        performances[key] = modelDevelop.trainModel(scaledTrain, trainLabels, scaledTest, testLabels)['f_micro'].tolist()
        if performances[key] >= bestCurrent:
            listOfSelectedFeatures = listOfCandidateFeatures.copy()
            bestCurrent = performances[key]
        elif bestCurrent > bestPerformance or len(listOfSelectedFeatures) == noOfFeatures:
            listOfSelectedFeatures.insert(0,selectedFeature)
            repeat = False

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
        for feature in listOfFeatures:
            key = feature
            scoreSet_train.clear()
            scoreSet_test.clear()
            for tweet in featureScores_train[feature]:
                scoreSet_train.append(tweet[2])
            for tweet in featureScores_test[feature]:
                scoreSet_test.append(tweet[2])
            for survivedFeature in listOfSelectedFeatures:
                key = key + " " + survivedFeature
                for index, tweet in enumerate(featureScores_train[survivedFeature]):
                    scoreSet_train[index].extend(tweet[2])
                for index, tweet in enumerate(featureScores_test[survivedFeature]):
                    scoreSet_test[index].extend(tweet[2])
            scalingModel = scaler.fit(scoreSet_train)
            scaledTrain = scalingModel.transform(scoreSet_train)
            scaledTest = scalingModel.transform(scoreSet_test)
            performances[key] = modelDevelop.trainModel(scaledTrain, trainLabels, scaledTest, testLabels)['f_micro'].tolist()
            if performances[key] > bestCurrent:
                bestCurrent = performances[key]
                candidateFeature = feature
        if bestCurrent <= bestPerformance or len(listOfSelectedFeatures) == noOfFeatures:
            repeat = False
        else:
            listOfSelectedFeatures.extend([candidateFeature])
            bestPerformance = bestCurrent
            listOfFeatures.remove(candidateFeature)

    ###############################################################################
    # Simplified Forward Selection ################################################
    ###############################################################################
#    
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
#            scoreSet_train.append(tweet[2])
#        for tweet in featureScores_test[feature]:
#            scoreSet_test.append(tweet[2])        
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
#            scoreSet_train.append(tweet[2])
#        for tweet in featureScores_test[selectedFeature]:
#            scoreSet_test.append(tweet[2])
#        for survivedFeature in listOfSelectedFeatures:
#            key = key + " " + survivedFeature
#            for index, tweet in enumerate(featureScores_train[survivedFeature]):
#                scoreSet_train[index].extend(tweet[2])
#            for index, tweet in enumerate(featureScores_test[survivedFeature]):
#                scoreSet_test[index].extend(tweet[2])
#        scalingModel = scaler.fit(scoreSet_train)
#        scaledTrain = scalingModel.transform(scoreSet_train)
#        scaledTest = scalingModel.transform(scoreSet_test)
#        performances[key] = modelDevelop.trainModel(scaledTrain, trainLabels, scaledTest, testLabels)['f_micro'].tolist()
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
#    
    ###############################################################################
    # Backward Selection ##########################################################
    ###############################################################################
    #
    scaler = preprocessing.Normalizer()
    performances = dict()
    scoreSet_train = []
    scoreSet_test = []
    listOfFeatures = [feature for feature in featureScores_train.keys()]
    iterator = 0
    bestPerformance = 0
    bestCurrent = 0
    droppedFeatures = []
    noOfFeatures = len(listOfFeatures)
    repeat = True
    while repeat:
        scoreSet_train.clear()
        scoreSet_test.clear()
        scoreSet_train = [[] for i in range(0, len(featureScores_train[listOfFeatures[0]]))]
        scoreSet_test = [[] for i in range(0, len(featureScores_test[listOfFeatures[0]]))]
        for feature in listOfFeatures:
            for index, tweet in enumerate(featureScores_train[feature]):
                scoreSet_train[index].extend(tweet[2])
            for index, tweet in enumerate(featureScores_test[feature]):
                scoreSet_test[index].extend(tweet[2])
        scalingModel = scaler.fit(scoreSet_train)
        scaledTrain = scalingModel.transform(scoreSet_train)
        scaledTest = scalingModel.transform(scoreSet_test)
        performances[iterator] = modelDevelop.trainModel(scaledTrain, trainLabels, scaledTest, testLabels)['f_micro'].tolist()
        if performances[iterator] >= bestPerformance:
            dropFeatureIndex = int(np.random.choice(len(listOfFeatures), 1))
            candidateFeature = [dropFeatureIndex, listOfFeatures[dropFeatureIndex]]
            droppedFeatures.append(candidateFeature)
            del listOfFeatures[dropFeatureIndex]
            bestPerformance = performances[iterator]
            iterator = iterator + 1
        else:
            listOfFeatures.insert(candidateFeature[0], candidateFeature[1])
            droppedFeatures.remove(candidateFeature)
            repeat = False
    print("best feature set is ")
    for feature in listOfFeatures:
        print(feature)
    #
    ###############################################################################
    # Baseline feature set ########################################################
    ###############################################################################

#     dataFrames = [pd.DataFrame([]) for i in range(1,23)]
#
#     # combination of all lexicons - train
#     featureScores_train["combLex"] = copy.deepcopy(featureScores_train["featureElv"])
#     for index, tweetScore in enumerate(featureScores_train["featureTlv"]):
#         featureScores_train["combLex"][index][2].extend(tweetScore[2])
#         featureScores_train["combLex"][index][2].extend(featureScores_train["featureTrt"][index][2])
#
#     # combination of all lexicons - test
#     featureScores_test["combLex"] = copy.deepcopy(featureScores_test["featureElv"])
#     for index, tweetScore in enumerate(featureScores_test["featureTlv"]):
#         featureScores_test["combLex"][index][2].extend(tweetScore[2])
#         featureScores_test["combLex"][index][2].extend(featureScores_test["featureTrt"][index][2])
#
#     # combination of all lexicons - train
#     featureScores_train["allLexicons"] = copy.deepcopy(featureScores_train["featureOne"])
#     for index, tweetScore in enumerate(featureScores_train["featureTwo"]):
#         featureScores_train["allLexicons"][index][2].extend(tweetScore[2])
#         featureScores_train["allLexicons"][index][2].extend(featureScores_train["featureThr"][index][2])
#         featureScores_train["allLexicons"][index][2].extend(featureScores_train["featureFor"][index][2])
#         featureScores_train["allLexicons"][index][2].extend(featureScores_train["featureFiv"][index][2])
#         featureScores_train["allLexicons"][index][2].extend(featureScores_train["featureSix"][index][2])
#         featureScores_train["allLexicons"][index][2].extend(featureScores_train["featureSev"][index][2])
#         featureScores_train["allLexicons"][index][2].extend(featureScores_train["featureEit"][index][2])
#         featureScores_train["allLexicons"][index][2].extend(featureScores_train["featureNin"][index][2])
#         featureScores_train["allLexicons"][index][2].extend(featureScores_train["featureTen"][index][2])
#         featureScores_train["allLexicons"][index][2].extend(featureScores_train["featureElv"][index][2])
#         featureScores_train["allLexicons"][index][2].extend(featureScores_train["featureTlv"][index][2])
#         featureScores_train["allLexicons"][index][2].extend(featureScores_train["featureTrt"][index][2])
#         featureScores_train["allLexicons"][index][2].extend(featureScores_train["featureFrt"][index][2])
#
#     # combination of all lexicons - test
#     featureScores_test["allLexicons"] = copy.deepcopy(featureScores_test["featureOne"])
#     for index, tweetScore in enumerate(featureScores_test["featureTwo"]):
#         featureScores_test["allLexicons"][index][2].extend(tweetScore[2])
#         featureScores_test["allLexicons"][index][2].extend(featureScores_test["featureThr"][index][2])
#         featureScores_test["allLexicons"][index][2].extend(featureScores_test["featureFor"][index][2])
#         featureScores_test["allLexicons"][index][2].extend(featureScores_test["featureFiv"][index][2])
#         featureScores_test["allLexicons"][index][2].extend(featureScores_test["featureSix"][index][2])
#         featureScores_test["allLexicons"][index][2].extend(featureScores_test["featureSev"][index][2])
#         featureScores_test["allLexicons"][index][2].extend(featureScores_test["featureEit"][index][2])
#         featureScores_test["allLexicons"][index][2].extend(featureScores_test["featureNin"][index][2])
#         featureScores_test["allLexicons"][index][2].extend(featureScores_test["featureTen"][index][2])
#         featureScores_test["allLexicons"][index][2].extend(featureScores_test["featureElv"][index][2])
#         featureScores_test["allLexicons"][index][2].extend(featureScores_test["featureTlv"][index][2])
#         featureScores_test["allLexicons"][index][2].extend(featureScores_test["featureTrt"][index][2])
#         featureScores_test["allLexicons"][index][2].extend(featureScores_test["featureFrt"][index][2])
#
#     # combination of all features - train
#     featureScores_train["allFeatures"] = copy.deepcopy(featureScores_train["featureOne"])
#     for index, tweetScore in enumerate(featureScores_train["featureTwo"]):
#         featureScores_train["allFeatures"][index][2].extend(tweetScore[2])
#         featureScores_train["allFeatures"][index][2].extend(featureScores_train["featureThr"][index][2])
#         featureScores_train["allFeatures"][index][2].extend(featureScores_train["featureFor"][index][2])
#         featureScores_train["allFeatures"][index][2].extend(featureScores_train["featureFiv"][index][2])
#         featureScores_train["allFeatures"][index][2].extend(featureScores_train["featureSix"][index][2])
#         featureScores_train["allFeatures"][index][2].extend(featureScores_train["featureSev"][index][2])
#         featureScores_train["allFeatures"][index][2].extend(featureScores_train["featureEit"][index][2])
#         featureScores_train["allFeatures"][index][2].extend(featureScores_train["featureNin"][index][2])
#         featureScores_train["allFeatures"][index][2].extend(featureScores_train["featureTen"][index][2])
#         featureScores_train["allFeatures"][index][2].extend(featureScores_train["featureElv"][index][2])
#         featureScores_train["allFeatures"][index][2].extend(featureScores_train["featureTlv"][index][2])
#         featureScores_train["allFeatures"][index][2].extend(featureScores_train["featureTrt"][index][2])
#         featureScores_train["allFeatures"][index][2].extend(featureScores_train["featureFrt"][index][2])
#         featureScores_train["allFeatures"][index][2].extend(featureScores_train["selfDict"][index][2])
#         featureScores_train["allFeatures"][index][2].extend(featureScores_train["query"][index][2])
#         featureScores_train["allFeatures"][index][2].extend(featureScores_train["symbol"][index][2])
#         featureScores_train["allFeatures"][index][2].extend(featureScores_train["tfidf"][index][2])
#         featureScores_train["allFeatures"][index][2].extend(featureScores_train["cosine"][index][2])
#
#     # combination of all features - test
#     featureScores_test["allFeatures"] = copy.deepcopy(featureScores_test["featureOne"])
#     for index, tweetScore in enumerate(featureScores_test["featureTwo"]):
#         featureScores_test["allFeatures"][index][2].extend(tweetScore[2])
#         featureScores_test["allFeatures"][index][2].extend(featureScores_test["featureThr"][index][2])
#         featureScores_test["allFeatures"][index][2].extend(featureScores_test["featureFor"][index][2])
#         featureScores_test["allFeatures"][index][2].extend(featureScores_test["featureFiv"][index][2])
#         featureScores_test["allFeatures"][index][2].extend(featureScores_test["featureSix"][index][2])
#         featureScores_test["allFeatures"][index][2].extend(featureScores_test["featureSev"][index][2])
#         featureScores_test["allFeatures"][index][2].extend(featureScores_test["featureEit"][index][2])
#         featureScores_test["allFeatures"][index][2].extend(featureScores_test["featureNin"][index][2])
#         featureScores_test["allFeatures"][index][2].extend(featureScores_test["featureTen"][index][2])
#         featureScores_test["allFeatures"][index][2].extend(featureScores_test["featureElv"][index][2])
#         featureScores_test["allFeatures"][index][2].extend(featureScores_test["featureTlv"][index][2])
#         featureScores_test["allFeatures"][index][2].extend(featureScores_test["featureTrt"][index][2])
#         featureScores_test["allFeatures"][index][2].extend(featureScores_test["featureFrt"][index][2])
#         featureScores_test["allFeatures"][index][2].extend(featureScores_test["selfDict"][index][2])
#         featureScores_test["allFeatures"][index][2].extend(featureScores_test["query"][index][2])
#         featureScores_test["allFeatures"][index][2].extend(featureScores_test["symbol"][index][2])
#         featureScores_test["allFeatures"][index][2].extend(featureScores_test["tfidf"][index][2])
#         featureScores_test["allFeatures"][index][2].extend(featureScores_test["cosine"][index][2])
#
#     scaler = preprocessing.Normalizer()
#     performances = dict()
#     scoreSet_train = []
#     scoreSet_test = []
#     listOfFeatures = [feature for feature in featureScores_train.keys()]
#     iterator = 0
#     bestPerformance = 0
#     bestCurrent = 0
#     noOfFeatures = len(listOfFeatures)
#
#     for feature_index, feature in enumerate(listOfFeatures):
#         scoreSet_train.clear()
#         scoreSet_test.clear()
#         for tweet in featureScores_train[feature]:
#             scoreSet_train.append(tweet[2])
#         for tweet in featureScores_test[feature]:
#             scoreSet_test.append(tweet[2])
#         scalingModel = scaler.fit(scoreSet_train)
#         scaledTrain = scalingModel.transform(scoreSet_train)
#         scaledTest = scalingModel.transform(scoreSet_test)
#         modelResult = modelDevelop.trainModel(scaledTrain, trainLabels, scaledTest, testLabels)
#         result = []
#         result.extend(modelResult['precision'])
#         result.extend(modelResult['recall'])
#         result.extend(modelResult['f_macro'])
#         result.extend([modelResult['f_micro']])
#         # A = copy.deepcopy()
#         dataFrames[feature_index] = dataFrames[feature_index].append(pd.DataFrame(result).T, ignore_index=True)
#
# writer = pd.ExcelWriter(r"C:\Users\Ali\Desktop\test-2.xlsx", engine='xlsxwriter')
# for index, feature in enumerate(dataFrames):
#     feature.to_excel(writer, sheet_name=listOfFeatures[index])
#
# writer.save()
# writer.close()
#
# # os.system("shutdown /s /t 1")
#
#
# #
# #    allFeatureDataFrame = pd.DataFrame([microAllFeature], columns=[i for i in range(1, 31)])
# #    allLexDataFrame = pd.DataFrame([microAllLex], columns=[i for i in range(1, 31)])
# #    microAllFeaturesDataFrame = microAllFeaturesDataFrame.append(allFeatureDataFrame, ignore_index=True)
# #    microAllLexDataFrame = microAllLexDataFrame.append(allLexDataFrame, ignore_index=True)
# #    microAllFeature.clear()
# #    microAllLex.clear()
# #    print("size " + str(size) +" finished")
# #
# # microAllLexDataFrame.to_csv(r"C:\Users\Ali\Desktop\fear-allLexiconMicroFscore-ManyIteration.csv", index=True, header=True, sep="\t")
# # microAllFeaturesDataFrame.to_csv(r"C:\Users\Ali\Desktop\fear-allFeaturesMicroFscore-ManyIteration.csv", index=True, header=True, sep="\t")