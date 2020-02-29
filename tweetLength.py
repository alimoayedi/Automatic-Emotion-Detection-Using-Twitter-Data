import sys
sys.path.append(r"D:\Synced Folder\PyCharm Codes")

import TestTrainFiles
import Tokenizer
import pandas as pd

testTrainFiles = TestTrainFiles.TestTrainFiles()
tokenizer = Tokenizer.Tokenizer()

sizeOfTweets=[]
trainTweets=[]

trainFile = testTrainFiles.getFile('train')[0]
with open(trainFile[0], 'r', errors="surrogateescape") as file:
    for line in file:
        fields = line.split('\t')
        fields[2] = fields[2].replace("\udc8d", "")
        fields[2] = fields[2].replace("\udc9d", "")
        trainTweets.append(fields)

# make a list of tweets class and their size after tokenization
for tweet in trainTweets:
    termsInTweets = tokenizer.tokenize(tweet[2], 'simple')
    sizeOfTweets.append(len(termsInTweets))

sizeOfTweets.sort()

A=pd.DataFrame(sizeOfTweets)
    
sum(sizeOfTweets)/len(sizeOfTweets)
    