import random


class ClassifierSelection:
    def __init__(self):
        self.votes = []

    def tweetVoting(self, predictions, weights=None, votingType=None):
        if weights is None:
            return self._majorityVoting(self, predictions)
        elif votingType == 'classifierwise':
            return self._classifierWiseVoting(self, predictions, weights)
        elif votingType == 'levelwise':
            return self._levelWiseVoting(self, predictions, weights)
        else:
            print("Error in votingType. It can be None, classifiedwise or levelwise.")

    @staticmethod
    def _levelWiseVoting(self, predictions, weights):
        listOfFeatures = [key for key in predictions.keys()]
        self.votes = [[0, 0, 0, 0] for item in range(0, len(predictions.get(listOfFeatures[0])))]
        for classifier in listOfFeatures:
            classifierPredictions = predictions.get(classifier).tolist()
            for index, predict in enumerate(classifierPredictions):
                self.votes[index][int(predict)] = self.votes[index][int(predict)] + (1 * weights.get(classifier)[int(predict)])
        return self.votes

    @staticmethod
    def _classifierWiseVoting(self, predictions, weights):
        listOfFeatures = [key for key in predictions.keys()]
        self.votes = [[0, 0, 0, 0] for item in range(0, len(predictions.get(listOfFeatures[0])))]
        for classifier in listOfFeatures:
            classifierPredictions = predictions.get(classifier).tolist()
            for index, predict in enumerate(classifierPredictions):
                self.votes[index][int(predict)] = self.votes[index][int(predict)] + (1 * weights.get(classifier))
        return self.votes

    @staticmethod
    def _majorityVoting(self, predictions):
        listOfFeatures = [key for key in predictions.keys()]
        self.votes = [[0, 0, 0, 0] for item in range(0, len(predictions.get(listOfFeatures[0])))]
        for classifier in listOfFeatures:
            classifierPredictions = predictions.get(classifier).tolist()
            for index, predict in enumerate(classifierPredictions):
                self.votes[index][int(predict)] = self.votes[index][int(predict)] + 1
        return self.votes

    def getPredictedLabels(self):
        predicts = []
        for tweet in self.votes:
            maximumVotes = [index for index, value in enumerate(tweet) if value == max(tweet)]
            predicts.append(str(maximumVotes[random.randint(0, len(maximumVotes) - 1)]))
        return predicts
