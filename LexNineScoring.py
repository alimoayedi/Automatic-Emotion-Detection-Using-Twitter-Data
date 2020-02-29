import sys
sys.path.append(r"D:\Synced Folder\PyCharm Codes")

import Lexicons
import TestTrainFiles
import Tokenizer
import TweetScoringVectorize as TSV

testTrainFiles = TestTrainFiles.TestTrainFiles()
lexicons = Lexicons.Lexicons()
tokenizer = Tokenizer.Tokenizer()

trainTweets = []

trainFile = testTrainFiles.getFile('test')[0]
trainTweets.clear()
with open(trainFile[0], 'r', errors="surrogateescape") as file:
    for line in file:
        fields = line.split('\t')
        fields[2] = fields[2].replace("\udc8d", "")
        fields[2] = fields[2].replace("\udc9d", "")
        trainTweets.append(fields)

dictionaryOfScoredTweets = []

for lexiconURL in lexicons.getLexicons('anger'):
    listOfScoresByLexicons = []  # list of scores by each lexicon
    scoring = TSV.tweetscoring(lexiconURL[0])
    for index, tweet in enumerate(trainTweets):
        termsInTweets = tokenizer.tokenize(tweet[2], 'simple')
        result = scoring.getScores(termsInTweets, lexiconURL[1])

        dictionaryOfScoredTweets.append(result)
        print(index + 1)
        print(len(termsInTweets))
        print("====")
        
with open(r"c:\users\ali\desktop\sadness-scores.csv", 'w') as f:
    for feature in dictionaryOfScoredTweets:
        for item in feature:
            f.write("%s\t" % item)
        f.write("\n")