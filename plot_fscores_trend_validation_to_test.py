import csv
from matplotlib import pyplot as plot

data=[]
stage=[]
lables = []
xTickes = []

with open(r"C:\Users\Ali\Desktop\micro_fs.csv") as micro:
    reader = csv.reader(micro)
    for value in reader:
        stage.append([int(value[2]),int(value[0])])
        lables.append(value[4])
        data.append([float(value[3]),float(value[1])])

for num in range(1,3):
    xTickes.extend([num])
    
figure, sub = plot.subplots(1, 1, sharex=True)

for i in range(0, len(data)):
    sub.plot(['validation', 'test'], data[i])
plot.legend(lables,ncol=2, bbox_to_anchor=(1, 1.025), loc=2)
ax = plot.gca()
ax.set_ylim([0.34, 0.48])
#ax.yaxis.grid(which="major", color='lightgray', linestyle='--', linewidth=1)
plot.ylabel('micro f-score')
plot.title('Emotion: Sadness')

plot.savefig(r'C:\Users\Ali\Desktop\Sadness.png', format='png', bbox_inches='tight', dpi=1200)

