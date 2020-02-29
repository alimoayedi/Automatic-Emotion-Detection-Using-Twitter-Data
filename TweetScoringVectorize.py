import numpy as np


class tweetscoring(object):

    def __init__(self, filename):
        self.lexicon = self._loadLexicon(filename)  # lexicon type "dictionary"

    def getScores(self, termsInTweets: object, lexiconType: str):
        if lexiconType == "binary":
            tweetTokenScore = self._scoreByLexicons_binary(self, termsInTweets)
        elif lexiconType == "decimal":
            tweetTokenScore = self._scoreByLexicons_decimal(self, termsInTweets)
        elif lexiconType == "psweighted":
            tweetTokenScore = self._scoreByLexicons_psweighted(self, termsInTweets)
        elif lexiconType == "categorized":
            tweetTokenScore = self._scoreByLexicons_categorized(self, termsInTweets)
        elif lexiconType == "biword":
            tweetTokenScore = self._scoreByLexicons_biword(self, termsInTweets)
        elif lexiconType == "pairword":
            tweetTokenScore = self._scoreByLexicons_pairword(self, termsInTweets)

        return tweetTokenScore

    @staticmethod
    def _loadLexicon(url: str) -> dict:
        lexiconFile = []
        with open(url, 'r', errors="ignore") as file:
            for line in file:
                term = line.split("\t")
                if term[-1].find("\n"):
                    term[-1] = term[-1][:-1]
                # print(term)
                lexiconFile.append(term)
        lexiconDict = {}
        for term in lexiconFile:
            # print(term)
            lexiconDict[term[1]] = term[2]

        return lexiconDict

    @staticmethod
    def _scoreByLexicons_decimal(self, listOfTokens) -> object:
        listOfScores = []
        # score terms that are already exists in lexicons.
        for term in listOfTokens:
            if term in self.lexicon:
                listOfScores.append(float(self.lexicon[term]))
            else:
                listOfScores.append(0)

        return listOfScores

    @staticmethod
    def _scoreByLexicons_binary(self, listOfTokens) -> object:
        listOfScores = []
        for term in listOfTokens:
            if term in self.lexicon:
                listOfScores.append(float(self.lexicon[term]))
            else:
                listOfScores.append(0)

        return listOfScores

    @staticmethod
    def _scoreByLexicons_psweighted(self, listOfTokens) -> object:
        listOfScores = []
        for term in listOfTokens:
            if term in self.lexicon:
                listOfScores.append(float(self.lexicon[term]))
            else:
                listOfScores.append(0)

        return listOfScores

    @staticmethod
    def _scoreByLexicons_categorized(self, listOfTokens) -> object:
        listOfScores = []
        for term in listOfTokens:
            if term in self.lexicon:
                listOfScores.append(float(self.lexicon[term]))
            else:
                listOfScores.append(0)

        return listOfScores

    @staticmethod
    def _scoreByLexicons_biword(self, listOfTokens) -> object:
        listOfScores = []
        for index in range(len(listOfTokens) - 1):
            biword = listOfTokens[index] + " " + listOfTokens[index + 1]
            if biword in self.lexicon:
                listOfScores.append(float(self.lexicon[biword]))
            else:
                listOfScores.append(0)

        return listOfScores

    @staticmethod
    def _scoreByLexicons_pairword(self, listOfTokens) -> object:
        listOfScores = np.zeros(len(listOfTokens))
        listFirstHalfMatch = []
        for key in self.lexicon.keys():
            for index in range(len(listOfTokens) - 1):
                if key.split('---')[0] == listOfTokens[index]:
                    listFirstHalfMatch.append([index, key])

        if len(listFirstHalfMatch) > 0:
            for item in listFirstHalfMatch:
                # single second term
                for subIndex in range(item[0] + 1, len(listOfTokens)):
                    # print('1' + item[1].split('---')[1] +'\t\t'+ listOfTokens[subIndex])
                    if item[1].split('---')[1] == listOfTokens[subIndex]:
                        # print('OK1')
                        listOfScores[item[0]] = listOfScores[item[0]] + float(self.lexicon[item[1]])
                        break
                # pair second term
                if ' ' in item[1].split('---')[1]:
                    for subIndex in range(item[0], len(listOfTokens) - 1):
                        # print('2' + item[1].split('---')[1] +'\t\t'+ listOfTokens[subIndex] + ' ' + listOfTokens[subIndex + 1])
                        if item[1].split('---')[1] == listOfTokens[subIndex] + ' ' + listOfTokens[subIndex + 1]:
                            # print('OK2')
                            listOfScores[item[0]] = listOfScores[item[0]] + float(self.lexicon[item[1]])
                            break

        # bi-word first half
        listFirstHalfMatch.clear()
        for key in self.lexicon.keys():
            for index in range(len(listOfTokens) - 2):
                if key.split('---')[0] == (listOfTokens[index] + ' ' + listOfTokens[index + 1]):
                    listFirstHalfMatch.append([index, key])

        if len(listFirstHalfMatch) > 0:
            for item in listFirstHalfMatch:
                # single second term
                for subIndex in range(item[0] + 1, len(listOfTokens)):
                    # print('3' + item[1].split('---')[1] +'\t\t'+ listOfTokens[subIndex])
                    if item[1].split('---')[1] == listOfTokens[subIndex]:
                        # print('OK3')
                        listOfScores[item[0]] = listOfScores[item[0]] + float(self.lexicon[item[1]])
                        break
                # pair second term
                if ' ' in item[1].split('---')[1]:
                    for subIndex in range(item[0] + 1, len(listOfTokens) - 1):
                        # print('4' + item[1].split('---')[1] +'\t\t'+ listOfTokens[subIndex])
                        if item[1].split('---')[1] == (listOfTokens[subIndex] + ' ' + listOfTokens[subIndex + 1]):
                            # print('OK4')
                            listOfScores[item[0]] = listOfScores[item[0]] + float(self.lexicon[item[1]])
                            break
        return listOfScores.tolist()