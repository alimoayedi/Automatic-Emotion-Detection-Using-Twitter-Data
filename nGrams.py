import numpy as np
import nltk
import Tokenizer


class nGrams:
    def __init__(self, n):
        self.corpusOfnGrams = []
        self.nGrams = n

    def trainCorpus(self, tweetsCorpus):
        tokenizer = Tokenizer.Tokenizer()
        for tweet in tweetsCorpus:
            termsInTweets = tokenizer.tokenize(tweet, 'simple')
            self.corpusOfnGrams.extend(list(nltk.ngrams(termsInTweets, self.nGrams)).copy())

    def getScoreVector(self, tokenizedTweet):
        score = np.zeros(len(self.corpusOfnGrams))
        nGramTweet = nltk.ngrams(tokenizedTweet, self.nGrams)
        for ngram in self.corpusOfnGrams:
            if ngram in nGramTweet:
                score[self.corpusOfnGrams.index(ngram)] = score[self.corpusOfnGrams.index(ngram)] + 1
        return score
