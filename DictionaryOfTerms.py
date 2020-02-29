import pandas as pd


class DictionaryOfTerms:
    def __init__(self):
        self.termsDict = dict()
        self.termScoreDict = {}

    def addToTermScoreDict(self, listOfTerms: list, listOfScores: list):
        for index, term in enumerate(listOfTerms):
            if term in self.termScoreDict.keys():
                self.termScoreDict[term][0] = self.termScoreDict[term][0] + 1
                self.termScoreDict[term][1] = self.termScoreDict[term][1] + listOfScores[index]
            else:
                self.termScoreDict[term] = [1, listOfScores[index]]

    def addToTermDict(self, listOfTerms: list):
        for term in listOfTerms:
            if term in self.termsDict.keys():
                self.termsDict[term] = self.termsDict[term] + 1
            else:
                self.termsDict[term] = 1

    def getTermScoreDict(self, percentage=100):
        termScoreDF = pd.DataFrame.from_dict(self.termScoreDict, orient='index')
        termScoreDF = termScoreDF.sort_values(0, ascending=False)
        return termScoreDF.head(int(len(termScoreDF) * (percentage / 100)))

    def getTermDict(self, percentage=100):
        termsDF = pd.DataFrame.from_dict(self.termsDict, orient='index')
        termsDF = termsDF.sort_values(0, ascending=False)
        return termsDF.head(int(len(termsDF) * (percentage / 100)))

