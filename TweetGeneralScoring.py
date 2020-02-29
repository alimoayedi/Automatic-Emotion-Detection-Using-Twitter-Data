import Lexicons
import Tokenizer
import TfDfCounter
import Word2VecModeling
import TweetScoringVectorize as TSV
import iDfCalculator
import GetCosineSimilarity
import DictionaryOfTerms
import QueryTermsScoring
import SymbolEffect
import RandomTermSelection

import pandas as pd

import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing


###############################################################################
# DEFINED FUNCTIONS ###########################################################
###############################################################################


class TweetGeneralScoring:
    def __init__(self):
        # initiate classes
        self.lexicons = Lexicons.Lexicons()
        self.tokenizer = Tokenizer.Tokenizer()
        self.symbolEffect = SymbolEffect.SymbolEffect()
        self.word2vecModeling = Word2VecModeling.Word2VecModel()
        self.getCosineScore = GetCosineSimilarity.GetCosineSimilarity()
        self.queryTermsScoring = QueryTermsScoring.QueryTermsScoring()
        self.dictClassZero = DictionaryOfTerms.DictionaryOfTerms()
        self.dictClassOne = DictionaryOfTerms.DictionaryOfTerms()
        self.dictClassTwo = DictionaryOfTerms.DictionaryOfTerms()
        self.dictClassThr = DictionaryOfTerms.DictionaryOfTerms()
        self.rndSelection = RandomTermSelection.RandomTermSelection()

        # initiate variables
        self.trainTweets = []
        self.trainTokenizedTweet = []
        self.testTweets = []
        self.testTokenizedTweet = []
        self.classOneTweetToken = []
        self.classTwoTweetToken = []
        self.classThrTweetToken = []
        self.classForTweetToken = []
        self.tfidf_scores = dict()

    def fit(self, trainFile):
        with open(trainFile[0], 'r', errors="surrogateescape") as file:
            for line in file:
                fields = line.split('\t')
                fields[2] = fields[2].replace("\udc8d", "")
                fields[2] = fields[2].replace("\udc9d", "")
                self.trainTweets.append(fields)
        for tweet in self.trainTweets:
            self.trainTokenizedTweet.append([tweet[0], tweet[1], self.tokenizer.tokenize(tweet[2], 'simple')])

        for tweet in self.trainTokenizedTweet:
            if tweet[1] == '0':
                self.dictClassZero.addToTermDict(listOfTerms=tweet[2])
            elif tweet[1] == '1':
                self.dictClassOne.addToTermDict(listOfTerms=tweet[2])
            elif tweet[1] == '2':
                self.dictClassTwo.addToTermDict(listOfTerms=tweet[2])
            else:
                self.dictClassThr.addToTermDict(listOfTerms=tweet[2])

    def fitTest(self, testFile):
        with open(testFile[0], 'r', errors="surrogateescape") as file:
            for line in file:
                fields = line.split('\t')
                fields[2] = fields[2].replace("\udc8d", "")
                fields[2] = fields[2].replace("\udc9d", "")
                self.testTweets.append(fields)
            for tweet in self.testTweets:
                self.testTokenizedTweet.append([tweet[0], tweet[1], self.tokenizer.tokenize(tweet[2], 'simple')])

    def getLabels(self, phase):
        labels = []
        if phase == 'train':
            for tweet in self.trainTokenizedTweet:
                labels.append(tweet[1])
        elif phase == 'test':
            for tweet in self.testTokenizedTweet:
                labels.append(tweet[1])
        return labels

    # Lexicon Scores ##############################################################
    def lexiconScoring(self, phase, tweetSize, lexiconURL):
        if phase == 'train':
            dataset = self.trainTokenizedTweet
        elif phase == 'test':
            dataset = self.testTokenizedTweet

        dictionaryOfScoredTweets = [[0, 0, []] for i in range(len(dataset))]
        listOfScoresByLexicons = []  # list of scores by each lexicon
        scoring = TSV.tweetscoring(lexiconURL[0])
        for index, tweet in enumerate(dataset):
            dictionaryOfScoredTweets[index][0] = tweet[0]
            dictionaryOfScoredTweets[index][1] = tweet[1]
            result = scoring.getScores(tweet[2], lexiconURL[1])

            while len(result) < tweetSize:
                result.append(0)

            if len(result) > tweetSize:
                result = self.rndSelection.randomSelection(scores=result, c=tweetSize, mode='simple')

            listOfScoresByLexicons.append(result)

        for index, score in enumerate(listOfScoresByLexicons):
            dictionaryOfScoredTweets[index][2].extend(score)

        listOfScoresByLexicons.clear()
        return dictionaryOfScoredTweets

    # Lexicon Nine Scores #########################################################
    def lexiconNineScoring(self, phase, lexiconNineAdd, tweetSize):
        if phase == 'train':
            dataset = self.trainTokenizedTweet
        elif phase == 'test':
            dataset = self.testTokenizedTweet
        lexNineScore = []
        lexNinTrainScores = pd.read_csv(lexiconNineAdd, index_col=0)
        for index, scoreVector in lexNinTrainScores.iterrows():
            score = scoreVector.iloc[:].dropna().values.tolist()

            while len(score) < tweetSize:
                score.append(0)
            if len(score) > tweetSize:
                score = self.rndSelection.randomSelection(scores=score, c=tweetSize, mode='simple')
            lexNineScore.append([dataset[index][0], dataset[index][1], score])
        return lexNineScore

    # Train tf-idf Scores #########################################################
    def tfidfScoring(self, phase):
        if phase == 'train':
            tfidfScores = []

            for tweet in self.trainTokenizedTweet:
                if tweet[1] == '0':  # tweets from each class are kept separately for TFIDF scores
                    self.classOneTweetToken.append(tweet[2])
                elif tweet[1] == '1':
                    self.classTwoTweetToken.append(tweet[2])
                elif tweet[1] == '2':
                    self.classThrTweetToken.append(tweet[2])
                else:
                    self.classForTweetToken.append(tweet[2])

            tfidfCalculator = iDfCalculator.IdfCalculator()
            # TFiDF training for class 0
            frequencyAnalyse = TfDfCounter.TfDfCounter(self.classOneTweetToken).TfDf()  # tweets are tokenized in the method
            tfidfClassOne = tfidfCalculator.getTFIDF(docCount=frequencyAnalyse['DocCount'],
                                                     tf=frequencyAnalyse['tf'],
                                                     df=frequencyAnalyse['df'])
            frequencyAnalyse.clear()
            frequencyAnalyse = TfDfCounter.TfDfCounter(self.classTwoTweetToken).TfDf()  # tweets are tokenized in the method
            tfidfClassTwo = tfidfCalculator.getTFIDF(docCount=frequencyAnalyse['DocCount'],
                                                     tf=frequencyAnalyse['tf'],
                                                     df=frequencyAnalyse['df'])
            frequencyAnalyse.clear()
            frequencyAnalyse = TfDfCounter.TfDfCounter(self.classThrTweetToken).TfDf()  # tweets are tokenized in the method
            tfidfClassThr = tfidfCalculator.getTFIDF(docCount=frequencyAnalyse['DocCount'],
                                                     tf=frequencyAnalyse['tf'],
                                                     df=frequencyAnalyse['df'])
            frequencyAnalyse.clear()
            frequencyAnalyse = TfDfCounter.TfDfCounter(self.classForTweetToken).TfDf()  # tweets are tokenized in the method
            tfidfClassFor = tfidfCalculator.getTFIDF(docCount=frequencyAnalyse['DocCount'],
                                                     tf=frequencyAnalyse['tf'],
                                                     df=frequencyAnalyse['df'])

            self.tfidf_scores = {k: [tfidfClassOne.get(k, 0) if k in tfidfClassOne else 0,
                                     tfidfClassTwo.get(k, 0) if k in tfidfClassTwo else 0,
                                     tfidfClassThr.get(k, 0) if k in tfidfClassThr else 0,
                                     tfidfClassFor.get(k, 0) if k in tfidfClassFor else 0]
                                 for k in set(tfidfClassOne) | set(tfidfClassTwo) | set(tfidfClassThr) | set(tfidfClassFor)}

            for tweet in self.trainTokenizedTweet:
                tempScore = np.array([0, 0, 0, 0])
                for term in tweet[2]:
                    if term in self.tfidf_scores:
                        tempScore = tempScore + np.array(self.tfidf_scores.get(term))
                tfidfScores.append([tweet[0], tweet[1], tempScore.tolist()])

            # Clear history
            del frequencyAnalyse
            del tfidfClassOne
            del tfidfClassTwo
            del tfidfClassThr
            del tfidfClassFor
            return tfidfScores

        elif phase == 'test':
            tfidfScores_test = []
            for tweet in self.testTokenizedTweet:
                tempScore = np.array([0, 0, 0, 0])
                for term in tweet[2]:
                    if term in self.tfidf_scores:
                        tempScore = tempScore + np.array(self.tfidf_scores.get(term))
                tfidfScores_test.append([tweet[0], tweet[1], tempScore.tolist()])
            return tfidfScores_test

    # Train Cosine Similarity Scores ##############################################
    def cosineScoring(self, phase):
        if phase == 'train':
            cosineScores = []
            classOneModel = self.word2vecModeling.getModel(self.classOneTweetToken, 400, 2)
            classTwoModel = self.word2vecModeling.getModel(self.classTwoTweetToken, 400, 2)
            classThrModel = self.word2vecModeling.getModel(self.classThrTweetToken, 400, 2)
            classForModel = self.word2vecModeling.getModel(self.classForTweetToken, 400, 2)

            save = False
            if save:
                classOneModel.wv.save_word2vec_format(r'D:\Thesis\Thesis-CE\Phyton Program\Word2Vec Data Files\ClassOneModel.bin')
                classTwoModel.wv.save_word2vec_format(r'D:\Thesis\Thesis-CE\Phyton Program\Word2Vec Data Files\ClassTwoModel.bin')
                classThrModel.wv.save_word2vec_format(r'D:\Thesis\Thesis-CE\Phyton Program\Word2Vec Data Files\ClassThreeModel.bin')
                classForModel.wv.save_word2vec_format(r'D:\Thesis\Thesis-CE\Phyton Program\Word2Vec Data Files\ClassFourModel.bin')

            self.getCosineScore.fit(classOneModel, classTwoModel, classThrModel, classForModel)
            for tweet in self.trainTokenizedTweet:
                cosineScores.append([tweet[0], tweet[1], self.getCosineScore.getSimilarity(tweet[2])])
            return cosineScores
        elif phase == 'test':
            cosineScores_test = []
            for tweet in self.testTweets:
                cosineScores_test.append([tweet[0], tweet[1], self.getCosineScore.getSimilarity(tweet[2])])
            return cosineScores_test

    # SELF DICTIONARY SCORING #####################################################
    def selfDictScoring(self, phase, topTermsPerc, tweetSize):
        if phase == 'train':
            selfDict = []
            for tweet in self.trainTokenizedTweet:
                if tweet[1] == '0':
                    termDF = self.dictClassZero.getTermDict(percentage=topTermsPerc)
                elif tweet[1] == '1':
                    termDF = self.dictClassOne.getTermDict(percentage=topTermsPerc)
                elif tweet[1] == '2':
                    termDF = self.dictClassTwo.getTermDict(percentage=topTermsPerc)
                else:
                    termDF = self.dictClassThr.getTermDict(percentage=topTermsPerc)

                tempScore = []
                for term in tweet[2]:
                    if term in termDF.index:
                        tempScore.append(termDF.loc[term][0])
                    else:
                        tempScore.append(0)
                while len(tempScore) < tweetSize:
                    tempScore.append(0)
                if len(tempScore) > tweetSize:
                    tempScore = self.rndSelection.randomSelection(scores=tempScore, c=tweetSize, mode='simple')
                selfDict.append([tweet[0], tweet[1], tempScore])
            return selfDict
        elif phase == 'test':
            selfDict_test = []
            for tweet in self.testTokenizedTweet:
                if tweet[1] == '0':
                    termDF = self.dictClassZero.getTermDict(percentage=topTermsPerc)
                elif tweet[1] == '1':
                    termDF = self.dictClassOne.getTermDict(percentage=topTermsPerc)
                elif tweet[1] == '2':
                    termDF = self.dictClassTwo.getTermDict(percentage=topTermsPerc)
                else:
                    termDF = self.dictClassThr.getTermDict(percentage=topTermsPerc)

                tempScore = []
                for term in tweet[2]:
                    if term in termDF.index:
                        tempScore.append(termDF.loc[term][0])
                    else:
                        tempScore.append(0)
                while len(tempScore) < tweetSize:
                    tempScore.append(0)
                if len(tempScore) > tweetSize:
                    tempScore = self.rndSelection.randomSelection(scores=tempScore, c=tweetSize, mode='simple')
                selfDict_test.append([tweet[0], tweet[1], tempScore])
            return selfDict_test

    # QUERY TERMS SCORING #########################################################
    def queryTermScoring(self, phase, emotion):
        if phase == 'train':
            queryScores = []
            self.queryTermsScoring.fit(emotion)
            self.queryTermsScoring.labelQueryTerm(self.trainTweets)
            for tweet in self.trainTokenizedTweet:
                queryScores.append([tweet[0], tweet[1], self.queryTermsScoring.getScore(tweet[2])])
            return queryScores
        elif phase == 'test':
            queryScores_test = []
            for tweet in self.testTokenizedTweet:
                queryScores_test.append([tweet[0], tweet[1], self.queryTermsScoring.getScore(tweet[2])])
            return queryScores_test

    # SYMBOL EFFECT ###############################################################
    def symbolScoring(self, phase):
        if phase == 'train':
            symbolScores = []
            for tweet in self.trainTokenizedTweet:
                symbolScores.append([tweet[0], tweet[1], self.symbolEffect.getCount_occurrence(tweet[2], ['!', '?'])])
            return symbolScores
        elif phase == 'test':
            symbolScores_test = []
            for tweet in self.testTokenizedTweet:
                symbolScores_test.append([tweet[0], tweet[1], self.symbolEffect.getCount_occurrence(tweet[2], ['!', '?'])])
            return symbolScores_test
