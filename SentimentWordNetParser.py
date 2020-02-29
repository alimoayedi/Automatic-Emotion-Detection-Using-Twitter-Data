import re

trainTweets = []
dictionaryOfTerms = {}
path = r"C:\Users\Ali\Desktop\Book1.txt"
with open(path, 'r', encoding="utf8", errors="surrogateescape") as file:
    for line in file:
        fields = line.split('\t')
        setOfTerms = fields[3].split(' ')
        for term in setOfTerms:
            term = re.sub("#\d+\n|#\d+", "", term).lower()
            if term not in dictionaryOfTerms:
                dictionaryOfTerms[term] = [float(fields[1]) - float(fields[2]), 1]
            else:
                dictionaryOfTerms[term] = [dictionaryOfTerms[term][0] + float(fields[1]) - float(fields[2]),
                                           dictionaryOfTerms[term][1] + 1]

for key, value in dictionaryOfTerms.items():
    dictionaryOfTerms[key] = value[0] / value[1]

file = open(r"C:\Users\Ali\Desktop\SentiWordNet.txt", "w", errors="surrogateescape")
for key, value in dictionaryOfTerms.items():
    file.write("x\t" + key + "\t" + str(value))
    file.write('\n')
file.close()
