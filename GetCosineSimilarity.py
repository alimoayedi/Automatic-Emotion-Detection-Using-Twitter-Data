import numpy


class GetCosineSimilarity:

    def __init__(self):
        self.ModelOne = object()
        self.ModelTwo = object()
        self.ModelThree = object()
        self.ModelFour = object()

    def fit(self, classOneModel: object, classTwoModel: object, classThreeModel: object, classFourModel: object):
        self.ModelOne = classOneModel
        self.ModelTwo = classTwoModel
        self.ModelThree = classThreeModel
        self.ModelFour = classFourModel

    def getSimilarity(self, arrayOfTerms: list):
        try:
            print(self.ModelOne.wv.most_similar('joy'))
        except:
            pass
        try:
            print(self.ModelTwo.wv.most_similar('joy'))
        except:
            pass
        try:
            print(self.ModelThree.wv.most_similar('joy'))
        except:
            pass
        try:
            print(self.ModelFour.wv.most_similar('joy'))
        except:
            pass
        tempScore = numpy.array([0, 0, 0, 0])
        cosineScore = numpy.array([0, 0, 0, 0])
        for index in range(len(arrayOfTerms)):
            # class 0
            tempScore[0] = sum([self.getScoreModelOne(arrayOfTerms, index, -2) if (index - 2 >= 0) else 0,
                                self.getScoreModelOne(arrayOfTerms, index, -1) if (index - 1 >= 0) else 0,
                                self.getScoreModelOne(arrayOfTerms, index, 1) if (index + 1 < len(arrayOfTerms)) else 0,
                                self.getScoreModelOne(arrayOfTerms, index, 2) if (index + 2 < len(arrayOfTerms)) else 0])
            # class 1
            tempScore[1] = sum([self.getScoreModelTwo(arrayOfTerms, index, -2) if (index - 2 >= 0) else 0,
                                self.getScoreModelTwo(arrayOfTerms, index, -1) if (index - 1 >= 0) else 0,
                                self.getScoreModelTwo(arrayOfTerms, index, 1) if (index + 1 < len(arrayOfTerms)) else 0,
                                self.getScoreModelTwo(arrayOfTerms, index, 2) if (index + 2 < len(arrayOfTerms)) else 0])
            # class 2
            tempScore[2] = sum([self.getScoreModelThree(arrayOfTerms, index, -2) if (index - 2 >= 0) else 0,
                                self.getScoreModelThree(arrayOfTerms, index, -1) if (index - 1 >= 0) else 0,
                                self.getScoreModelThree(arrayOfTerms, index, 1) if (index + 1 < len(arrayOfTerms)) else 0,
                                self.getScoreModelThree(arrayOfTerms, index, 2) if (index + 2 < len(arrayOfTerms)) else 0])
            # class 3
            tempScore[3] = sum([self.getScoreModelFour(arrayOfTerms, index, -2) if (index - 2 >= 0) else 0,
                                self.getScoreModelFour(arrayOfTerms, index, -1) if (index - 1 >= 0) else 0,
                                self.getScoreModelFour(arrayOfTerms, index, 1) if (index + 1 < len(arrayOfTerms)) else 0,
                                self.getScoreModelFour(arrayOfTerms, index, 2) if (index + 2 < len(arrayOfTerms)) else 0])
            # sum up scores for a tweet
            cosineScore = cosineScore + tempScore

        return cosineScore.tolist()

    def getScoreModelOne(self, arrayOfTerms: list, indexOfTerm: int, window: int):
        try:
            return self.ModelOne.wv.similarity(arrayOfTerms[indexOfTerm], arrayOfTerms[indexOfTerm + window])
        except KeyError:
            return 0

    def getScoreModelTwo(self, arrayOfTerms: list, indexOfTerm: int, window: int):
        try:
            return self.ModelTwo.wv.similarity(arrayOfTerms[indexOfTerm], arrayOfTerms[indexOfTerm + window])
        except KeyError:
            return 0

    def getScoreModelThree(self, arrayOfTerms: list, indexOfTerm: int, window: int):
        try:
            return self.ModelThree.wv.similarity(arrayOfTerms[indexOfTerm], arrayOfTerms[indexOfTerm + window])
        except KeyError:
            return 0

    def getScoreModelFour(self, arrayOfTerms: list, indexOfTerm: int, window: int):
        try:
            return self.ModelFour.wv.similarity(arrayOfTerms[indexOfTerm], arrayOfTerms[indexOfTerm + window])
        except KeyError:
            return 0