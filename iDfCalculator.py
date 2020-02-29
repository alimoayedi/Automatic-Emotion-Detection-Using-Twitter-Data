import math


class IdfCalculator:
    def __init__(self):
        pass

    def getTFIDF(self, docCount: int, tf: dict, df: dict):
        return {k: tf.get(k, 0) * math.log10(docCount / df.get(k, 0)) for k in set(df)}
