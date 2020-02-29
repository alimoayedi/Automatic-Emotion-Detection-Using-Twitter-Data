import sys
sys.path.append(r"D:\Synced Folder\PyCharm Codes")
# import os

import Lexicons
import TestTrainFiles
import Tokenizer
import StopList
import TfDfCounter
import Word2VecModeling
import TweetScoringVectorize as TSV
import iDfCalculator
import GetCosineSimilarity
import DictionaryOfTerms
import QueryTermsScoring
import SymbolEffect
import RandomTermSelection
import ClassifierSelection

import pandas as pd

import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing

###############################################################################
# DEFINED FUNCTIONS ###########################################################
###############################################################################
microAllFeaturesDataFrame = pd.DataFrame([])
microAllLexDataFrame = pd.DataFrame([])


if __name__ == '__main__':
    # functions
    testTrainFiles = TestTrainFiles.TestTrainFiles()
    lexicons = Lexicons.Lexicons()
    stopList = StopList.StopList()
    tokenizer = Tokenizer.Tokenizer()
    getCosineScore = GetCosineSimilarity.GetCosineSimilarity()
    queryTermsScoring = QueryTermsScoring.QueryTermsScoring()
    symbolScoring = SymbolEffect.SymbolEffect()
    word2vecModeling = Word2VecModeling.Word2VecModel()
    dictClassZero = DictionaryOfTerms.DictionaryOfTerms()
    dictClassOne = DictionaryOfTerms.DictionaryOfTerms()
    dictClassTwo = DictionaryOfTerms.DictionaryOfTerms()
    dictClassThr = DictionaryOfTerms.DictionaryOfTerms()
    rndSelection = RandomTermSelection.RandomTermSelection()
    clsSelection = ClassifierSelection.ClassifierSelection()

    classOneTweetToken = []
    classTwoTweetToken = []
    classThrTweetToken = []
    classForTweetToken = []

    # List of scores given to each tweet for each emotion (Data type: Dictionary)
    dictionaryOfScoredTweets = []
    dictionaryOfScoredTweets_test = []

    cosineScores = []
    cosineScores_test = []

    tfidfScores = []
    tfidfScores_test = []

    selfDict = []
    selfDict_test = []

    queryScores = []
    queryScores_test = []

    symbolScores = []
    symbolScores_test = []

    labels = []
    labels_test = []

    # Variable to keep a list of tweets imported from file
    trainTweets = []
    testTweets = []

    scores_dict = {
        "lexOneScores": [],
        "lexTwoScores": [],
        "lexThrScores": [],
        "lexForScores": [],
        "lexFivScores": [],
        "lexSixScores": [],
        "lexSevScores": [],
        "lexEitScores": [],
        "lexNinScores": [],
        "lexTenScores": [],
        "lexElvScores": [],
        "lexTlvScores": [],
        "lexTrtScores": [],
        "lexComScores": [],
        "lexBinScores": [],
        "lexAllScores": [],
        "selfLexScores": [],
        "queryScores": [],
        "symbolEffect": [],
        "tf_idfScores": [],
        "cosineScores": [],
        "AllScores": [],
        "RFS": [],
        "FS" : [],
        "SFS": [],
        "BS": []}

    scores_test_dict = {
        "lexOneScores": [],
        "lexTwoScores": [],
        "lexThrScores": [],
        "lexForScores": [],
        "lexFivScores": [],
        "lexSixScores": [],
        "lexSevScores": [],
        "lexEitScores": [],
        "lexNinScores": [],
        "lexTenScores": [],
        "lexElvScores": [],
        "lexTlvScores": [],
        "lexTrtScores": [],
        "lexComScores": [],
        "lexBinScores": [],
        "lexAllScores": [],
        "selfLexScores": [],
        "queryScores": [],
        "symbolEffect": [],
        "tf_idfScores": [],
        "cosineScores": [],
        "AllScores": [],
        "RFS": [],
        "FS" : [],
        "SFS": [],
        "BS": []}

    predicts_dict = {
        "lexOneScores": [],
        "lexTwoScores": [],
        "lexThrScores": [],
        "lexForScores": [],
        "lexFivScores": [],
        "lexSixScores": [],
        "lexSevScores": [],
        "lexEitScores": [],
        "lexNinScores": [],
        "lexTenScores": [],
        "lexElvScores": [],
        "lexTlvScores": [],
        "lexTrtScores": [],
        "lexComScores": [],
        "lexBinScores": [],
        "lexAllScores": [],
        "selfLexScores": [],
        "queryScores": [],
        "symbolEffect": [],
        "tf_idfScores": [],
        "cosineScores": [],
        "AllScores": [],
        "RFS": [],
        "FS" : [],
        "SFS": [],
        "BS": []}

    ###############################################################################
    # Customize parameters for each emotion #######################################
    ###############################################################################

    trainFile = testTrainFiles.getFile('train')[0]

    # set tweet size based on emotion
    if trainFile[1] == 'anger':
        normalize = True    # checked
        tweetSize = 33      # selected by checking micro f-scores for all lexicons. (33)
        topTermsPerc = 70   # selected by checking micro f-scores (70).
        modelIteration = 100000
        lexNineScoreAdd = r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\lexNineScores\anger_lexNineScore_train.csv"
        #develop
        # lexNineScoreAdd_test = r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\lexNineScores\anger_lexNineScore_dev.csv"
        #test
        lexNineScoreAdd_test = r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\lexNineScores\anger_lexNineScore_test.csv"
    elif trainFile[1] == 'joy':
        normalize = True    # checked, actually they don't differ
        tweetSize = 33      # selected by checking micro f-scores for all lexicons. (33)
        topTermsPerc = 35   # selected by checking micro f-scores (35).
        modelIteration = 100000
        lexNineScoreAdd = r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\lexNineScores\joy_lexNineScore_train.csv"
        #develop
        # lexNineScoreAdd_test = r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\lexNineScores\joy_lexNineScore_dev.csv"
        #test
        lexNineScoreAdd_test = r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\lexNineScores\joy_lexNineScore_test.csv"
    elif trainFile[1] == 'fear':
        normalize = True    # checked, no improve in micro f_scores
        tweetSize = 36      # selected by checking micro f-scores for all lexicons. (36)
        topTermsPerc = 20    # selected by checking micro f-scores (20).
        modelIteration = 100000
        lexNineScoreAdd = r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\lexNineScores\fear_lexNineScore_train.csv"
        #develop
        # lexNineScoreAdd_test = r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\lexNineScores\fear_lexNineScore_dev.csv"
        #test
        lexNineScoreAdd_test = r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\lexNineScores\fear_lexNineScore_test.csv"
    else:
        normalize = True    # checked, individual performance increase however micro f_Score of all features decreases
        tweetSize = 27      # selected by checking micro f-scores for all lexicons. (27)
        topTermsPerc = 35   # selected by checking micro f-scores (35).
        modelIteration = 100000
        lexNineScoreAdd = r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\lexNineScores\sadness_lexNineScore_train.csv"
        #develop
        # lexNineScoreAdd_test = r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\lexNineScores\sadness_lexNineScore_dev.csv"
        #test
        lexNineScoreAdd_test = r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\lexNineScores\sadness_lexNineScore_test.csv"

    ###############################################################################
    # Train tf-idf Scores #########################################################
    ###############################################################################

    trainTweets.clear()
    with open(trainFile[0], 'r', errors="surrogateescape") as file:
        for line in file:
            fields = line.split('\t')
            fields[2] = fields[2].replace("\udc8d", "")
            fields[2] = fields[2].replace("\udc9d", "")
            termsInTweets = tokenizer.tokenize(fields[2], 'simple')
            if fields[1] == '0':  # tweets from each class are kept separately for TFIDF scores
                classOneTweetToken.append(termsInTweets)
                # if scores are going to be considered 'add'(following) command (for all levels of emotion) should be moved inside
                # scoring loop over lexicons, before checking the length of scores.
                dictClassZero.addToTermDict(listOfTerms=termsInTweets)
            elif fields[1] == '1':
                classTwoTweetToken.append(termsInTweets)
                dictClassOne.addToTermDict(listOfTerms=termsInTweets)
            elif fields[1] == '2':
                classThrTweetToken.append(termsInTweets)
                dictClassTwo.addToTermDict(listOfTerms=termsInTweets)
            else:
                classForTweetToken.append(termsInTweets)
                dictClassThr.addToTermDict(listOfTerms=termsInTweets)

    tfidfCalculator = iDfCalculator.IdfCalculator()
    # TFiDF training for class 0
    frequencyAnalyse = TfDfCounter.TfDfCounter(classOneTweetToken).TfDf()  # tweets are tokenized in the method
    tfidfClassOne = tfidfCalculator.getTFIDF(docCount=frequencyAnalyse['DocCount'],
                                             tf=frequencyAnalyse['tf'],
                                             df=frequencyAnalyse['df'])
    frequencyAnalyse.clear()
    frequencyAnalyse = TfDfCounter.TfDfCounter(classTwoTweetToken).TfDf()  # tweets are tokenized in the method
    tfidfClassTwo = tfidfCalculator.getTFIDF(docCount=frequencyAnalyse['DocCount'],
                                             tf=frequencyAnalyse['tf'],
                                             df=frequencyAnalyse['df'])
    frequencyAnalyse.clear()
    frequencyAnalyse = TfDfCounter.TfDfCounter(classThrTweetToken).TfDf()  # tweets are tokenized in the method
    tfidfClassThr = tfidfCalculator.getTFIDF(docCount=frequencyAnalyse['DocCount'],
                                             tf=frequencyAnalyse['tf'],
                                             df=frequencyAnalyse['df'])
    frequencyAnalyse.clear()
    frequencyAnalyse = TfDfCounter.TfDfCounter(classForTweetToken).TfDf()  # tweets are tokenized in the method
    tfidfClassFor = tfidfCalculator.getTFIDF(docCount=frequencyAnalyse['DocCount'],
                                             tf=frequencyAnalyse['tf'],
                                             df=frequencyAnalyse['df'])

    tfidf_scores = {k: [tfidfClassOne.get(k, 0) if k in tfidfClassOne else 0,
                        tfidfClassTwo.get(k, 0) if k in tfidfClassTwo else 0,
                        tfidfClassThr.get(k, 0) if k in tfidfClassThr else 0,
                        tfidfClassFor.get(k, 0) if k in tfidfClassFor else 0]
                    for k in set(tfidfClassOne) | set(tfidfClassTwo) | set(tfidfClassThr) | set(tfidfClassFor)}

    # Clear history
    del tfidfClassOne
    del tfidfClassTwo
    del tfidfClassThr
    del tfidfClassFor

    ###############################################################################
    # Train Cosine Similarity Scores ##############################################
    ###############################################################################

    classOneModel = word2vecModeling.getModel(classOneTweetToken, 400, 2)
    classTwoModel = word2vecModeling.getModel(classTwoTweetToken, 400, 2)
    classThrModel = word2vecModeling.getModel(classThrTweetToken, 400, 2)
    classForModel = word2vecModeling.getModel(classForTweetToken, 400, 2)

    save = False
    if save:
        classOneModel.wv.save_word2vec_format(r'D:\Thesis\Thesis-CE\Phyton Program\Word2Vec Data Files\ClassOneModel.bin')
        classTwoModel.wv.save_word2vec_format(r'D:\Thesis\Thesis-CE\Phyton Program\Word2Vec Data Files\ClassTwoModel.bin')
        classThrModel.wv.save_word2vec_format(r'D:\Thesis\Thesis-CE\Phyton Program\Word2Vec Data Files\ClassThreeModel.bin')
        classForModel.wv.save_word2vec_format(r'D:\Thesis\Thesis-CE\Phyton Program\Word2Vec Data Files\ClassFourModel.bin')

    ###############################################################################
    # Train Lexicon Scores ########################################################
    ###############################################################################

    trainFile = testTrainFiles.getFile('train')[0]
    trainTweets.clear()
    with open(trainFile[0], 'r', errors="surrogateescape") as file:
        for line in file:
            fields = line.split('\t')
            fields[2] = fields[2].replace("\udc8d", "")
            fields[2] = fields[2].replace("\udc9d", "")
            trainTweets.append(fields)

    dictionaryOfScoredTweets = [[0, 0, []] for i in range(len(trainTweets))]

    for lexiconURL in lexicons.getLexicons(trainFile[1]):
        listOfScoresByLexicons = []  # list of scores by each lexicon
        scoring = TSV.tweetscoring(lexiconURL[0])
        for index, tweet in enumerate(trainTweets):
            dictionaryOfScoredTweets[index][0] = tweet[0]
            dictionaryOfScoredTweets[index][1] = tweet[1]
            termsInTweets = tokenizer.tokenize(tweet[2], 'simple')
            result = scoring.getScores(termsInTweets, lexiconURL[1])

            while len(result) < tweetSize:
                result.append(0)

            if len(result) > tweetSize:
                result = rndSelection.randomSelection(scores=result, c=tweetSize, mode='simple')

            listOfScoresByLexicons.append(result)

        for index, score in enumerate(listOfScoresByLexicons):
            dictionaryOfScoredTweets[index][2].append(score)

        listOfScoresByLexicons.clear()
    # Lexicon Nine (pair-wise) scoring ############################################
    lexNinTrainScores = pd.read_csv(lexNineScoreAdd, index_col=0)
    for index, scoreVector in lexNinTrainScores.iterrows():
        score = scoreVector.iloc[:].dropna().values.tolist()

        while len(score) < tweetSize:
            score.append(0)

        if len(score) > tweetSize:
            score = rndSelection.randomSelection(scores=score, c=tweetSize, mode='simple')

        dictionaryOfScoredTweets[index][2].insert(8, score)
    # TF_IDF SCORING ##############################################################
    for tweet in trainTweets:
        termsInTweets = tokenizer.tokenize(tweet[2], 'simple')
        tempScore = np.array([0, 0, 0, 0])
        for term in termsInTweets:
            if term in tfidf_scores:
                tempScore = tempScore + np.array(tfidf_scores.get(term))
        tfidfScores.append([tweet[0], tweet[1], tempScore.tolist()])
    # COSINE SCORING ##############################################################
    getCosineScore.fit(classOneModel, classTwoModel, classThrModel, classForModel)
    for tweet in trainTweets:
        termsInTweets = tokenizer.tokenize(tweet[2], 'simple')
        cosineScores.append([tweet[0], tweet[1], getCosineScore.getSimilarity(termsInTweets)])
    # SELF DICTIONARY SCORING #####################################################
    for tweet in trainTweets:
        termsInTweets = tokenizer.tokenize(tweet[2], 'simple')                
        scoreSet = []
        for level in range(0, 4):
            tempScore = []
            if level == 0:
                termDF = dictClassZero.getTermDict(percentage=topTermsPerc)
            elif level == 1:
                termDF = dictClassOne.getTermDict(percentage=topTermsPerc)
            elif level == 2:
                termDF = dictClassTwo.getTermDict(percentage=topTermsPerc)
            else:
                termDF = dictClassThr.getTermDict(percentage=topTermsPerc)

            for term in termsInTweets:
                if term in termDF.index:
                    tempScore.append(termDF.loc[term][0])
                else:
                    tempScore.append(0)
            while len(tempScore) < tweetSize:
                tempScore.append(0)                    
            if len(tempScore) > tweetSize:
                tempScore = rndSelection.randomSelection(scores=tempScore, c=tweetSize, mode='simple')
            scoreSet.extend(tempScore)
        selfDict.append([tweet[0], tweet[1], scoreSet])
    # QUERY TERMS SCORING #####################################################
    queryTermsScoring.fit(trainFile[1])
    queryTermsScoring.labelQueryTerm(trainTweets)
    for tweet in trainTweets:
        termsInTweets = tokenizer.tokenize(tweet[2], 'simple')
        queryScores.append([tweet[0], tweet[1], queryTermsScoring.getScore(termsInTweets)])
    # SYMBOL EFFECT ###########################################################
    for tweet in trainTweets:
        termsInTweets = tokenizer.tokenize(tweet[2], 'simple')
        symbolScores.append([tweet[0], tweet[1], symbolScoring.getCount_occurrence(termsInTweets, ['!', '?'])])

    print("########## Training Done!!!!!!!! ##########")

    ###############################################################################
    # Develop Model ###############################################################
    ###############################################################################

    # Extract Labels ##############################################################
    labels.clear()
    for score in dictionaryOfScoredTweets:
        labels.append(score[1])

    # Organize Scores #############################################################

    for index, score in enumerate(dictionaryOfScoredTweets):
        scores_dict.get("lexOneScores").append(score[2][0])
        scores_dict.get("lexTwoScores").append(score[2][1])
        scores_dict.get("lexThrScores").append(score[2][2])
        scores_dict.get("lexForScores").append(score[2][3])
        scores_dict.get("lexFivScores").append(score[2][4])
        scores_dict.get("lexSixScores").append(score[2][5])
        scores_dict.get("lexSevScores").append(score[2][6])
        scores_dict.get("lexEitScores").append(score[2][7])
        scores_dict.get("lexNinScores").append(score[2][8])
        scores_dict.get("lexTenScores").append(score[2][9])
        scores_dict.get("lexElvScores").append(score[2][10])
        scores_dict.get("lexTlvScores").append(score[2][11])
        scores_dict.get("lexTrtScores").append(score[2][12])
        # combination of Ratings_Warriner lexicon
        scores_dict.get("lexComScores").append(score[2][10].copy())
        scores_dict.get("lexComScores")[index].extend(score[2][11])
        scores_dict.get("lexComScores")[index].extend(score[2][12])
        # lexicon BingLiu, combination of unigram index 13 and bigram index 14
        scores_dict.get("lexBinScores").append(score[2][13].copy())
        scores_dict.get("lexBinScores")[index].extend(score[2][14])
        # combination of all lexicons
        scores_dict.get("lexAllScores").append([value for tweet_score in score[2] for value in tweet_score])
        # other features
        scores_dict.get("selfLexScores").append(selfDict[index][2])
        scores_dict.get("queryScores").append(queryScores[index][2])
        scores_dict.get("symbolEffect").append(symbolScores[index][2])
        scores_dict.get("tf_idfScores").append(tfidfScores[index][2])
        scores_dict.get("cosineScores").append(cosineScores[index][2])
        scores_dict.get("AllScores").append([value for tweet_score in score[2] for value in tweet_score])
        scores_dict.get("AllScores")[index].extend(tfidfScores[index][2])
        scores_dict.get("AllScores")[index].extend(cosineScores[index][2])
        scores_dict.get("AllScores")[index].extend(selfDict[index][2])
        scores_dict.get("AllScores")[index].extend(queryScores[index][2])
        scores_dict.get("AllScores")[index].extend(symbolScores[index][2])
        
        scores_dict.get("RFS").append([value for index, tweet_score in enumerate(score[2]) if index in [0,2] for value in tweet_score])
        # scores_dict.get("RFS")[index].extend(queryScores[index][2])
        
        scores_dict.get("FS").append([value for index, tweet_score in enumerate(score[2]) if index in [0,2] for value in tweet_score])
        # scores_dict.get("FS")[index].extend(queryScores[index][2])
        
        scores_dict.get("SFS").append([value for index, tweet_score in enumerate(score[2]) if index in [2] for value in tweet_score])
        
        scores_dict.get("BS").append([value for index, tweet_score in enumerate(score[2]) if index not in [10] for value in tweet_score])
        #scores_dict.get("BS")[index].extend(selfDict[index][2])
        #scores_dict.get("BS")[index].extend(tfidfScores[index][2])
        
        
        
    # Define kernels ##############################################################

    LexOneModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    LexTwoModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    LexThrModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    LexForModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    LexFivModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    LexSixModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    LexSevModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    LexEitModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    LexNinModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    LexTenModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    LexElvModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    LexTlvModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    LexTrtModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    LexComModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    LexBinModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    LexAllModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    SelfLexModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    QueryLexModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    SymbolEfFModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    TfidfModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    CosineScoreModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    AllModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    
    RFSModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    FSModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    SFSModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)
    BSModel = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=modelIteration)

    # Scale Data ##################################################################

    scaler = preprocessing.Normalizer()

    scaledLexOne = scaler.fit(scores_dict["lexOneScores"])
    scaledLexTwo = scaler.fit(scores_dict["lexTwoScores"])
    scaledLexThr = scaler.fit(scores_dict["lexThrScores"])
    scaledLexFor = scaler.fit(scores_dict["lexForScores"])
    scaledLexFiv = scaler.fit(scores_dict["lexFivScores"])
    scaledLexSix = scaler.fit(scores_dict["lexSixScores"])
    scaledLexSev = scaler.fit(scores_dict["lexSevScores"])
    scaledLexEit = scaler.fit(scores_dict["lexEitScores"])
    scaledLexNin = scaler.fit(scores_dict["lexNinScores"])
    scaledLexTen = scaler.fit(scores_dict["lexTenScores"])
    scaledLexElv = scaler.fit(scores_dict["lexElvScores"])
    scaledLexTlv = scaler.fit(scores_dict["lexTlvScores"])
    scaledLexTrt = scaler.fit(scores_dict["lexTrtScores"])
    scaledLexCom = scaler.fit(scores_dict["lexComScores"])
    scaledLexBin = scaler.fit(scores_dict["lexBinScores"])
    scaledLexAll = scaler.fit(scores_dict["lexAllScores"])
    scaledSelfLex = scaler.fit(scores_dict["selfLexScores"])
    scaledQueryLex = scaler.fit(scores_dict["queryScores"])
    scaledSymbol = scaler.fit(scores_dict["symbolEffect"])
    scaledTfidf = scaler.fit(scores_dict["tf_idfScores"])
    scaledCosine = scaler.fit(scores_dict["cosineScores"])
    scaledAll = scaler.fit(scores_dict["AllScores"])
    
    scaledRFS = scaler.fit(scores_dict["RFS"])
    scaledFS = scaler.fit(scores_dict["FS"])
    scaledSFS = scaler.fit(scores_dict["SFS"])
    scaledBS = scaler.fit(scores_dict["BS"])

    # Fit Model ###################################################################
    if normalize:
        LexOneModel.fit(scaledLexOne.transform(scores_dict["lexOneScores"]), labels)
        LexTwoModel.fit(scaledLexTwo.transform(scores_dict["lexTwoScores"]), labels)
        LexThrModel.fit(scaledLexThr.transform(scores_dict["lexThrScores"]), labels)
        LexForModel.fit(scaledLexFor.transform(scores_dict["lexForScores"]), labels)
        LexFivModel.fit(scaledLexFiv.transform(scores_dict["lexFivScores"]), labels)
        LexSixModel.fit(scaledLexSix.transform(scores_dict["lexSixScores"]), labels)
        LexSevModel.fit(scaledLexSev.transform(scores_dict["lexSevScores"]), labels)
        LexEitModel.fit(scaledLexEit.transform(scores_dict["lexEitScores"]), labels)
        LexNinModel.fit(scaledLexNin.transform(scores_dict["lexNinScores"]), labels)
        LexTenModel.fit(scaledLexTen.transform(scores_dict["lexTenScores"]), labels)
        LexElvModel.fit(scaledLexElv.transform(scores_dict["lexElvScores"]), labels)
        LexTlvModel.fit(scaledLexTlv.transform(scores_dict["lexTlvScores"]), labels)
        LexTrtModel.fit(scaledLexTrt.transform(scores_dict["lexTrtScores"]), labels)
        LexComModel.fit(scaledLexCom.transform(scores_dict["lexComScores"]), labels)
        LexBinModel.fit(scaledLexBin.transform(scores_dict["lexBinScores"]), labels)
        LexAllModel.fit(scaledLexAll.transform(scores_dict["lexAllScores"]), labels)
        SelfLexModel.fit(scaledSelfLex.transform(scores_dict["selfLexScores"]), labels)
        QueryLexModel.fit(scaledQueryLex.transform(scores_dict["queryScores"]), labels)
        SymbolEfFModel.fit(scaledSymbol.transform(scores_dict["symbolEffect"]), labels)
        TfidfModel.fit(scaledTfidf.transform(scores_dict["tf_idfScores"]), labels)
        CosineScoreModel.fit(scaledCosine.transform(scores_dict["cosineScores"]), labels)
        AllModel.fit(scaledAll.transform(scores_dict["AllScores"]), labels)
        
        RFSModel.fit(scaledAll.transform(scores_dict["RFS"]), labels)
        FSModel.fit(scaledAll.transform(scores_dict["FS"]), labels)
        SFSModel.fit(scaledAll.transform(scores_dict["SFS"]), labels)
        BSModel.fit(scaledAll.transform(scores_dict["BS"]), labels)

    ###############################################################################
    # Develop test file ###########################################################
    ###############################################################################

    testTweets.clear()
    testFile = testTrainFiles.getFile('test')[0]
    # Open and read file
    with open(testFile[0], 'r', errors="surrogateescape") as source:
        for line in source:
            fields = line.split('\t')
            fields[2] = fields[2].replace("\udc8d", "")
            fields[2] = fields[2].replace("\udc9d", "")
            testTweets.append(fields)

    dictionaryOfScoredTweets_test = [[0, 0, []] for i in range(len(testTweets))]

    for lexiconURL in lexicons.getLexicons(testFile[1]):
        listOfScoresByLexicons = []  # list of scores by each lexicon
        scoring = TSV.tweetscoring(lexiconURL[0])
        for index, tweet in enumerate(testTweets):
            dictionaryOfScoredTweets_test[index][0] = tweet[0]
            dictionaryOfScoredTweets_test[index][1] = tweet[1]
            termsInTweets = tokenizer.tokenize(tweet[2], 'simple')
            result = scoring.getScores(termsInTweets, lexiconURL[1])

            while len(result) < tweetSize:
                result.append(0)

            if len(result) > tweetSize:
                result = rndSelection.randomSelection(scores=result, c=tweetSize, mode='simple')

            listOfScoresByLexicons.append(result)

        for index, score in enumerate(listOfScoresByLexicons):
            dictionaryOfScoredTweets_test[index][2].append(score)

        listOfScoresByLexicons.clear()
    # Lexicon Nine (pair-wise) scoring ############################################
    lexNinTrainScores = pd.read_csv(lexNineScoreAdd_test, index_col=0)
    for index, scoreVector in lexNinTrainScores.iterrows():
        score = scoreVector.astype(float).iloc[:].dropna().values.tolist()

        while len(score) < tweetSize:
            score.append(0)

        if len(score) > tweetSize:
            score = rndSelection.randomSelection(scores=score, c=tweetSize, mode='simple')
            # randomTerms = np.random.choice(len(score), tweetSize, replace=False)
            # randomTerms = randomTerms.tolist()
            # randomTerms.sort()
            # score = [score[index] for index in randomTerms]
        dictionaryOfScoredTweets_test[index][2].insert(8, score)
    # TF_IDF SCORING ##############################################################
    for tweet in testTweets:
        termsInTweets = tokenizer.tokenize(tweet[2], 'simple')
        tempScore = np.array([0, 0, 0, 0])
        for term in termsInTweets:
            if term in tfidf_scores:
                tempScore = tempScore + np.array(tfidf_scores.get(term))
        tfidfScores_test.append([tweet[0], tweet[1], tempScore.tolist()])
    # COSINE SCORING ##############################################################
    for tweet in testTweets:
        termsInTweets = tokenizer.tokenize(tweet[2], 'simple')
        cosineScores_test.append([tweet[0], tweet[1], getCosineScore.getSimilarity(termsInTweets)])
    # SELF DICTIONARY SCORING #####################################################
    for tweet in testTweets:
        termsInTweets = tokenizer.tokenize(tweet[2], 'simple')
        scoreSet = []
        for level in range(0, 4):
            tempScore = []
            if level == 0:
                termDF = dictClassZero.getTermDict(percentage=topTermsPerc)
            elif level == 1:
                termDF = dictClassOne.getTermDict(percentage=topTermsPerc)
            elif level == 2:
                termDF = dictClassTwo.getTermDict(percentage=topTermsPerc)
            else:
                termDF = dictClassThr.getTermDict(percentage=topTermsPerc)

            for term in termsInTweets:
                if term in termDF.index:
                    tempScore.append(termDF.loc[term][0])
                else:
                    tempScore.append(0)
            
            while len(tempScore) < tweetSize:
                tempScore.append(0)                    
            if len(tempScore) > tweetSize:
                tempScore = rndSelection.randomSelection(scores=tempScore, c=tweetSize, mode='simple')
            scoreSet.extend(tempScore)
        selfDict_test.append([tweet[0], tweet[1], scoreSet])    

    # QUERY SCORING ##############################################################
    for tweet in testTweets:
        termsInTweets = tokenizer.tokenize(tweet[2], 'simple')
        queryScores_test.append([tweet[0], tweet[1], queryTermsScoring.getScore(termsInTweets)])
    # SYMBOL EFFECT ##############################################################
    for tweet in testTweets:
        termsInTweets = tokenizer.tokenize(tweet[2], 'simple')
        symbolScores_test.append([tweet[0], tweet[1], symbolScoring.getCount_occurrence(termsInTweets, ['!', '?'])])

    print("########## Testing scores checked!!!!!! ##########" + str(tweetSize))

    # Getting labels of test samples #############################################
    scoreSet_test = []

    for score in dictionaryOfScoredTweets_test:
        labels_test.append(score[1])

    ###############################################################################
    # Predictions #################################################################
    ###############################################################################

    for index, score in enumerate(dictionaryOfScoredTweets_test):
        scores_test_dict.get("lexOneScores").append(score[2][0])
        scores_test_dict.get("lexTwoScores").append(score[2][1])
        scores_test_dict.get("lexThrScores").append(score[2][2])
        scores_test_dict.get("lexForScores").append(score[2][3])
        scores_test_dict.get("lexFivScores").append(score[2][4])
        scores_test_dict.get("lexSixScores").append(score[2][5])
        scores_test_dict.get("lexSevScores").append(score[2][6])
        scores_test_dict.get("lexEitScores").append(score[2][7])
        scores_test_dict.get("lexNinScores").append(score[2][8])
        scores_test_dict.get("lexTenScores").append(score[2][9])
        scores_test_dict.get("lexElvScores").append(score[2][10])
        scores_test_dict.get("lexTlvScores").append(score[2][11])
        scores_test_dict.get("lexTrtScores").append(score[2][12])
        # combination of Ratings_Warriner lexicon
        scores_test_dict.get("lexComScores").append(score[2][10].copy())
        scores_test_dict.get("lexComScores")[index].extend(score[2][11])
        scores_test_dict.get("lexComScores")[index].extend(score[2][12])
        # lexicon BingLiu combination of unigram index 13 and bigram index 14
        scores_test_dict.get("lexBinScores").append(score[2][13].copy())
        scores_test_dict.get("lexBinScores")[index].extend(score[2][14])
        # combination of all lexicons
        scores_test_dict.get("lexAllScores").append([value for tweet_score in score[2] for value in tweet_score])
        # other features
        scores_test_dict.get("selfLexScores").append(selfDict_test[index][2])
        scores_test_dict.get("queryScores").append(queryScores_test[index][2])
        scores_test_dict.get("symbolEffect").append(symbolScores_test[index][2])
        scores_test_dict.get("tf_idfScores").append(tfidfScores_test[index][2])
        scores_test_dict.get("cosineScores").append(cosineScores_test[index][2])
        scores_test_dict.get("AllScores").append([value for tweet_score in score[2] for value in tweet_score])
        scores_test_dict.get("AllScores")[index].extend(tfidfScores_test[index][2])
        scores_test_dict.get("AllScores")[index].extend(cosineScores_test[index][2])
        scores_test_dict.get("AllScores")[index].extend(selfDict_test[index][2])
        scores_test_dict.get("AllScores")[index].extend(queryScores_test[index][2])
        scores_test_dict.get("AllScores")[index].extend(symbolScores_test[index][2])
        
        scores_test_dict.get("RFS").append([value for index, tweet_score in enumerate(score[2]) if index in [0, 2] for value in tweet_score])
        #scores_test_dict.get("RFS")[index].extend(queryScores_test[index][2])
        
        scores_test_dict.get("FS").append([value for index, tweet_score in enumerate(score[2]) if index in [0, 2] for value in tweet_score])
        #scores_test_dict.get("FS")[index].extend(queryScores_test[index][2])
        
        scores_test_dict.get("SFS").append([value for index, tweet_score in enumerate(score[2]) if index in [2] for value in tweet_score])
        
        scores_test_dict.get("BS").append([value for index, tweet_score in enumerate(score[2]) if index not in [10] for value in tweet_score])
        #scores_test_dict.get("BS")[index].extend(selfDict_test[index][2])
        #scores_test_dict.get("BS")[index].extend(tfidfScores_test[index][2])
        

    if normalize:
        predicts_dict["lexOneScores"] = LexOneModel.predict(scaledLexOne.transform(scores_test_dict["lexOneScores"]))
        predicts_dict["lexTwoScores"] = LexTwoModel.predict(scaledLexTwo.transform(scores_test_dict["lexTwoScores"]))
        predicts_dict["lexThrScores"] = LexThrModel.predict(scaledLexThr.transform(scores_test_dict["lexThrScores"]))
        predicts_dict["lexForScores"] = LexForModel.predict(scaledLexFor.transform(scores_test_dict["lexForScores"]))
        predicts_dict["lexFivScores"] = LexFivModel.predict(scaledLexFiv.transform(scores_test_dict["lexFivScores"]))
        predicts_dict["lexSixScores"] = LexSixModel.predict(scaledLexSix.transform(scores_test_dict["lexSixScores"]))
        predicts_dict["lexSevScores"] = LexSevModel.predict(scaledLexSev.transform(scores_test_dict["lexSevScores"]))
        predicts_dict["lexEitScores"] = LexEitModel.predict(scaledLexEit.transform(scores_test_dict["lexEitScores"]))
        predicts_dict["lexNinScores"] = LexNinModel.predict(scaledLexNin.transform(scores_test_dict["lexNinScores"]))
        predicts_dict["lexTenScores"] = LexTenModel.predict(scaledLexTen.transform(scores_test_dict["lexTenScores"]))
        predicts_dict["lexElvScores"] = LexElvModel.predict(scaledLexElv.transform(scores_test_dict["lexElvScores"]))
        predicts_dict["lexTlvScores"] = LexTlvModel.predict(scaledLexTlv.transform(scores_test_dict["lexTlvScores"]))
        predicts_dict["lexTrtScores"] = LexTrtModel.predict(scaledLexTrt.transform(scores_test_dict["lexTrtScores"]))
        predicts_dict["lexComScores"] = LexComModel.predict(scaledLexCom.transform(scores_test_dict["lexComScores"]))
        predicts_dict["lexBinScores"] = LexBinModel.predict(scaledLexBin.transform(scores_test_dict["lexBinScores"]))
        predicts_dict["lexAllScores"] = LexAllModel.predict(scaledLexAll.transform(scores_test_dict["lexAllScores"]))
        predicts_dict["selfLexScores"] = SelfLexModel.predict(scaledSelfLex.transform(scores_test_dict["selfLexScores"]))
        predicts_dict["queryScores"] = QueryLexModel.predict(scaledQueryLex.transform(scores_test_dict["queryScores"]))
        predicts_dict["symbolEffect"] = SymbolEfFModel.predict(scaledSymbol.transform(scores_test_dict["symbolEffect"]))
        predicts_dict["tf_idfScores"] = TfidfModel.predict(scaledTfidf.transform(scores_test_dict["tf_idfScores"]))
        predicts_dict["cosineScores"] = CosineScoreModel.predict(scaledCosine.transform(scores_test_dict["cosineScores"]))
        predicts_dict["AllScores"] = AllModel.predict(scaledAll.transform(scores_test_dict["AllScores"]))
        
        predicts_dict["RFS"] = RFSModel.predict(scaledRFS.transform(scores_test_dict["RFS"]))
        predicts_dict["FS"] = FSModel.predict(scaledFS.transform(scores_test_dict["FS"]))
        predicts_dict["SFS"] = SFSModel.predict(scaledSFS.transform(scores_test_dict["SFS"]))
        predicts_dict["BS"] = BSModel.predict(scaledBS.transform(scores_test_dict["BS"]))
        

    # Print results ################################################################

    linear_result = [
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexOneScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexTwoScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexThrScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexForScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexFivScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexSixScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexSevScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexEitScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexNinScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexTenScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexElvScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexTlvScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexTrtScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexComScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexBinScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexAllScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["selfLexScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["queryScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["symbolEffect"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["tf_idfScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["cosineScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["AllScores"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["RFS"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["FS"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["SFS"]),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["BS"])
                     ]

    micro_fscores = [
                     metrics.f1_score(labels_test, predicts_dict["lexOneScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["lexTwoScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["lexThrScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["lexForScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["lexFivScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["lexSixScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["lexSevScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["lexEitScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["lexNinScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["lexTenScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["lexElvScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["lexTlvScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["lexTrtScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["lexComScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["lexBinScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["lexAllScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["selfLexScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["queryScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["symbolEffect"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["tf_idfScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["cosineScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["AllScores"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["RFS"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["FS"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["SFS"], average='micro'),
                     metrics.f1_score(labels_test, predicts_dict["BS"], average='micro')]
    
    macro_fscores = [
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexOneScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexTwoScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexThrScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexForScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexFivScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexSixScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexSevScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexEitScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexNinScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexTenScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexElvScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexTlvScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexTrtScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexComScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexBinScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["lexAllScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["selfLexScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["queryScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["symbolEffect"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["tf_idfScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["cosineScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["AllScores"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["RFS"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["FS"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["SFS"],average='macro'),
                     metrics.precision_recall_fscore_support(labels_test, predicts_dict["BS"],average='macro')]


    

    ###############################################################################
    # Classifier Selection ########################################################
    ###############################################################################

    clsSelection.tweetVoting(predicts_dict, None)
    classifiedPredictions = clsSelection.getPredictedLabels()
    metrics.precision_recall_fscore_support(labels_test, classifiedPredictions)
    metrics.f1_score(labels_test, classifiedPredictions, average='micro')
    metrics.precision_recall_fscore_support(labels_test, classifiedPredictions,average='macro')
    
    
    ###############################################################################
    # Save Precision, Recall and F Scores(micro and macro) ########################
    ###############################################################################
    
#    featuresPerformance = []
#    for feature in linear_result:
#        for item in feature:
#            featuresPerformance.append(list(item))
#    featuresPerformance = pd.DataFrame(featuresPerformance)
#    featuresPerformance.to_csv(r"c:\users\ali\desktop\anger-performance-test data.csv", index=True, header=True, sep="\t")
#    
#    micro_fscores = pd.DataFrame(micro_fscores)
#    micro_fscores.to_csv(r"c:\users\ali\desktop\anger-micro scores-test data.csv", index=True, header=True, sep="\t")
        
    with open(r"c:\users\ali\desktop\sadness-test data-feature metrics.csv", 'w') as f:
        for feature in linear_result:
            for item in feature:
                f.write("%s\n" % item)
                
    with open(r"c:\users\ali\desktop\sadness-test data-macro metrics.csv", 'w') as f:
        for feature in macro_fscores:
            for item in feature:
                if item is not None:
                    f.write("%.3f\n" % item)
                
    with open(r"c:\users\ali\desktop\sadness-test data-micro f-scores.csv", 'w') as f:
        for feature in micro_fscores:
            f.write("%.3f\n" % feature)
    
    allscoresDataFrame = pd.DataFrame(scores_dict.get("AllScores"))
    allscoresDataFrame.to_csv(r"c:\users\ali\desktop\sadness-test data-train scores.csv", index=True, header=True, sep="\t")
    
    allscoresDataFrame_test = pd.DataFrame(scores_test_dict.get("AllScores"))
    allscoresDataFrame_test.to_csv(r"c:\users\ali\desktop\sadness-test data-test scores.csv", index=True, header=True, sep="\t")    
    
    
    ###############################################################################
    # Shutdown PC after completion by force to close ##############################
    ###############################################################################
    
    # os.system("shutdown /s /t 1")
    
    ###############################################################################
    # Repetition ##################################################################
    ###############################################################################
    
    # microAllFeature.append(metrics.f1_score(labels_test, predicts_dict["AllScores"], average='micro'))
    # microAllLex.append(metrics.f1_score(labels_test, predicts_dict["lexAllScores"], average='micro'))
    #
    # allFeatureDataFrame = pd.DataFrame([microAllFeature], columns=[i for i in range(3, 5)])
    # allLexDataFrame = pd.DataFrame([microAllLex], columns=[i for i in range(3, 5)])
    # microAllFeaturesDataFrame = microAllFeaturesDataFrame.append(allFeatureDataFrame, ignore_index=True)
    # microAllLexDataFrame = microAllLexDataFrame.append(allLexDataFrame, ignore_index=True)
    # microAllFeature.clear()
    # microAllLex.clear()
    # print("iteration " + str(iteration) + " done")
    # microAllLexDataFrame.to_csv("/users/ali/desktop/fear-allLexiconMicroFscore-ManyIteration.csv", index=True, header=True, sep="\t")
    # microAllFeaturesDataFrame.to_csv("/users/ali/desktop/fear-allFeaturesMicroFscore-ManyIteration.csv", index=True, header=True, sep="\t")
