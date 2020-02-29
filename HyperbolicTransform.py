from math import exp


class HyperbolicTransform:
    def __init__(self):
        pass

    def transform(self, listOfValues):
        mappedValues = []
        for value in listOfValues:
            if len(value) > 1:
                scores = []
                for score in value:
                    scores.append(self._hyperbolicFunc(score))
                mappedValues.append(scores)
            else:
                mappedValues.append(self._hyperbolicFunc(value))
        return mappedValues

    @staticmethod
    def _hyperbolicFunc(value):
        try:
            return (exp(value)-exp(-1*value))/(exp(value)+exp(-1*value))
        except OverflowError:
            if value > 0:
                return 1
            else:
                return -1
