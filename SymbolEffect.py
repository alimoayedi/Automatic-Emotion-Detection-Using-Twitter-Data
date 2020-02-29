import numpy as np


class SymbolEffect:
    def __init__(self):
        pass

    def getCount_occurrence(self, tokenizedTweet, symbolList):
        scores = np.zeros(2*len(symbolList))
        for term in tokenizedTweet:
            for index, symbol in enumerate(symbolList):
                if symbol in term:
                    occurrenceCount = term.count(symbol)
                    scores[index] = scores[index] + 1
                    scores[index + len(symbolList)] = scores[index + len(symbolList)] + occurrenceCount
        return scores.tolist()
