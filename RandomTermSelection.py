import numpy as np


class RandomTermSelection:
    def __init__(self):
        pass

    def randomSelection(self, scores, c=1, mode='simple'):
        if all(np.array(scores) == 0):
            mode = 'simple'
        if mode == 'simple':
            randomIndexes = np.random.choice(len(scores), c, replace=False)
            randomIndexes = randomIndexes.tolist()
            randomIndexes.sort()
            samples = [scores[index] for index in randomIndexes]
            return samples
        elif mode == 'weighted':
            absScores = [abs(item) for item in scores]
            weights = [value / sum(absScores) for value in absScores]
            if np.count_nonzero(weights) < c:
                randomIndexes = np.random.choice(len(weights), np.count_nonzero(weights), replace=False, p=weights).tolist()
                randomIndexes.sort()
                samples = [scores[index] for index in randomIndexes]
                samples.extend(np.zeros(c-np.count_nonzero(weights)).tolist())
                return samples
            else:
                randomIndexes = np.random.choice(len(weights), c, replace=False, p=weights).tolist()
                randomIndexes.sort()
                samples = [scores[index] for index in randomIndexes]
                return samples
