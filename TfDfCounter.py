import Tokenizer
import StopList


class TfDfCounter:

    def __init__(self, tweets):
        self.ListOfTweetsToken = tweets

    def TfDf(self):
        tf = dict()
        df = dict()
        DocCount = len(self.ListOfTweetsToken)
        stopwordList = StopList.StopList().getStopList(False)
        for tweet in self.ListOfTweetsToken:
            tempDictionary = {}
            tempDocFreq = {}
            for tweetToken in tweet:
                if tweetToken not in stopwordList:
                    if tweetToken not in tempDictionary:
                        tempDictionary[tweetToken] = 1
                        tempDocFreq[tweetToken] = 1
                    else:
                        tempDictionary[tweetToken] = tempDictionary[tweetToken] + 1
            tf = {k: tf.get(k, 0) + tempDictionary.get(k, 0) for k in set(tf) | set(tempDictionary)}
            df = {k: df.get(k, 0) + tempDocFreq.get(k, 0) for k in set(df) | set(tempDocFreq)}

        return {'DocCount': DocCount, 'tf': tf, 'df': df}
