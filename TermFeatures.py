from sklearn.feature_extraction.text import CountVectorizer


class TermFeatures:
    def __init__(self, learningCorpus):
        self.corpus = learningCorpus
        vectorizer = CountVectorizer()
        self.learnVocabSet = vectorizer.fit(self.corpus)

    def getTermFeatures(self, tweetCorpus):
        return (self.learnVocabSet.transform(tweetCorpus)).toarray()
