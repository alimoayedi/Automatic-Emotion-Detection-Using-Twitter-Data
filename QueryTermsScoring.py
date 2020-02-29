import Tokenizer
import numpy


class QueryTermsScoring:
    def __init__(self):
        self.query_list = {
            "anger": r"D:\Thesis\Thesis-CE\Phyton Program\QueryTerms\anger.txt",
            "joy": r"D:\Thesis\Thesis-CE\Phyton Program\QueryTerms\joy.txt",
            "fear": r"D:\Thesis\Thesis-CE\Phyton Program\QueryTerms\fear.txt",
            "sadness": r"D:\Thesis\Thesis-CE\Phyton Program\QueryTerms\sadness.txt"
        }
        self.queryDict = dict()

    def fit(self, emotion: str):
        with open(self.query_list[emotion], 'r', errors="surrogateescape") as file:
            for line in file:
                self.queryDict[line.strip('\n')] = []

    def labelQueryTerm(self, tweetsList):
        tokenizer = Tokenizer.Tokenizer()
        for tweet in tweetsList:
            termsInTweets = tokenizer.tokenize(tweet[2], 'simple')
            for term in termsInTweets:
                if term in list(self.queryDict.keys()):
                    self.queryDict[term].append(tweet[1])
        self._scoreQueryTerms()

    def getScore(self,tokenizedTweet, noOfClasses=4):   # in our work each tweet might be from one of the four possible classes
        scores = numpy.zeros(noOfClasses)
        for term in tokenizedTweet:
            if term in list(self.queryDict.keys()):
                for index, score in enumerate(self.queryDict[term]):
                    scores[index] = max(scores[index], score)
        return scores.tolist()

    def _scoreQueryTerms(self):
        for term in list(self.queryDict.keys()):
            labelList = numpy.array(self.queryDict[term])
            unique, counts = numpy.unique(labelList, return_counts=True)
            self.queryDict[term] = (counts/sum(counts)).tolist()







