import csv
import numpy
from matplotlib import pyplot as plot
import matplotlib.patches as mpatches

microScoresAllLex = []
microScoresAllFth = []
dictPerc = []
xTickes = []

with open(r"C:\Users\Ali\Desktop\sadness-all features micro f.csv") as micro:
    reader = csv.reader(micro)
    for value in reader:
        microScoresAllFth.append(value[1])

with open(r"C:\Users\Ali\Desktop\sadness-all lex micro F.csv") as trainSize:
    reader = csv.reader(trainSize)
    for value in reader:
        microScoresAllLex.append(value[1])

# with open(r"C:\Users\Ali\Desktop\SizeOfTweets_test_sadness.csv", 'r') as devSize:
#    reader = csv.reader(devSize)
#    for value in reader:
#        tweetSize_test.extend(value)

for num in range(3, 41):
    dictPerc.extend([num])

for num in range(3, 41, 3):
    xTickes.extend([num])

microScoresAllLex = numpy.array(microScoresAllLex)
microScoresAllFth = numpy.array(microScoresAllFth)

microScoresAllLex = microScoresAllLex.astype(numpy.float)
microScoresAllFth = microScoresAllFth.astype(numpy.float)

yellow_patch = mpatches.Patch(color='yellow', label='average length of tweets(train)')
green_patch = mpatches.Patch(color='green', label='95th percentile of tweet sizes(validation)')
red_patch = mpatches.Patch(color='red', label='95th percentile of tweet sizes(train)')
black_patch = mpatches.Patch(color='black', label='micro f-scores (all features)')
gray_patch = mpatches.Patch(color='gray', label='micro f-scores (all lexicons)')

figure, sub = plot.subplots(1, 1, sharex=True)
sub.plot(dictPerc, microScoresAllLex, c='gray')
sub.plot(dictPerc, microScoresAllFth, c='k')
plot.axvline(19.26, color='yellow', linestyle='--')
#plot.axvline(31.1, color='red', linestyle='--') # train
plot.axvline(32, color='red', linestyle='--')   # train
plot.axvline(31, color='green', linestyle='--') # test
#plot.xticks(xTickes)
plot.minorticks_on()
ax = plot.gca()
#ax.set_ylim([0.625, 0.825])
ax.set_xlim([3, 40])
plot.xlabel('tweet size\nTrain Data: max size: 159, min size: 2, Q95: 32'
            '\nValidation Data: max size: 41, min size: 2, Q95: 31')
plot.ylabel('micro f-score')
plot.title('Emotion: Sadness')
plot.legend(handles=[black_patch, gray_patch, red_patch, green_patch, yellow_patch], loc='best', ncol=1)

plot.savefig(r'C:\Users\Ali\Desktop\legend.png', format='png', bbox_inches='tight', dpi=1200)
#
