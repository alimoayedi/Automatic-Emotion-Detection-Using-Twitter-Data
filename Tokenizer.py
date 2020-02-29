import WebTokenizer
import re


class Tokenizer:
    def __init__(self):
        pass

    def tokenize(self, tweet: object, tokenization_type: str):
        if tokenization_type == 'web':
            return self.web_tokenization(self, tweet)
        elif tokenization_type == 'simple':
            return self.simple_tokenization(self, tweet)

    @staticmethod
    def web_tokenization(self, tweet: object):
        tokenizer = WebTokenizer.TweetTokenizer()
        dictionaryOfTerms = []
        newTokensList = tokenizer.tokenize(tweet)
        dictionaryOfTerms = self._createDictionary(newTokensList, dictionaryOfTerms)
        return dictionaryOfTerms

    @staticmethod
    def simple_tokenization(self, tweet: object):
        dictionaryOfTerms = []
        newTokensList = [x for x in re.split(r"([()/\n\"_\-.,!?&]+)?\s?", tweet) if x]
        dictionaryOfTerms = self._createDictionary(newTokensList, dictionaryOfTerms)
        return dictionaryOfTerms

    @staticmethod
    def _createDictionary(newTokensList: list, dictionaryOfTerms: object):
        for term in newTokensList:
            dictionaryOfTerms.append(term)
        return dictionaryOfTerms
