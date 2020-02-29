from sklearn import svm
from sklearn import metrics


class ModelDevelop:
    def __init__(self):
        self.predictedLabels = []
    

    def trainModel(self, trainData, trainLabels, testData, testLabel):
        model = svm.LinearSVC(tol=0.000001, random_state=None, max_iter=100000)
        model.fit(trainData, trainLabels)
        self.predictedLabels = model.predict(testData)

        performance = {'precision': metrics.precision_recall_fscore_support(testLabel, self.predictedLabels)[0].tolist(),
                       'recall': metrics.precision_recall_fscore_support(testLabel, self.predictedLabels)[1].tolist(),
                       'f_level': metrics.precision_recall_fscore_support(testLabel, self.predictedLabels)[2].tolist(),
                       'f_micro': metrics.f1_score(testLabel, self.predictedLabels, average='micro')
                       }
        return performance

    def getPredictedLabels(self):
        return self.predictedLabels






