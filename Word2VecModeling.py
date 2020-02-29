from gensim.models import word2vec


class Word2VecModel:
    def __init__(self):
        pass

    def getModel(self, dataset: list, size: int, window: int):
        model = word2vec.Word2Vec(dataset, size=size, window=window)
        return model
