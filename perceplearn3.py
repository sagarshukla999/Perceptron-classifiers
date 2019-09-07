import sys
import os
import re
import glob
from collections import Counter
import random
path = 'op_spam_training_data'


allpaths = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))
totalfeatures = Counter()
for i in allpaths:

    with open(i, 'r') as i:
        lines = i.read()
    lines = lines.lower()
    lines = re.sub(r"[^a-zA-Z]+", ' ', lines)
    arr = lines.split()
    count = Counter(arr)
    totalfeatures += count


filteredfeatures = {}
weights1 = {}
weights2 = {}
u1 = {}
u2 = {}

for word in totalfeatures:
    if totalfeatures[word] < 1200 and totalfeatures[word] > 6:
        filteredfeatures[word] = totalfeatures[word]
        weights1[word] = 0
        weights2[word] = 0
        u1[word] = 0
        u2[word] = 0


k = 0
bias1 = 0
bias2 = 0
beta1 = 0
beta2 = 0
c = 1

while k < 10:
    random.Random(k*3).shuffle(allpaths)
    for paths in allpaths:
        activation1 = 0
        activation2 = 0
        if "negative_polarity" in paths:
            y1 = -1
        else:
            y1 = 1
        if "deceptive_from_MTurk" in paths:
            y2 = -1
        else:
            y2 = 1
        with open(paths, 'r') as i:
            lines = i.read()
        lines = lines.lower()
        lines = re.sub(r"[^a-zA-Z]+", ' ', lines)
        arr = lines.split()
        count = Counter(arr)

        for word in count:
            if word in filteredfeatures:
                if weights1[word] != 0:
                    activation1 += weights1[word]*count[word]
                    activation2 += weights2[word]*count[word]

        activation1 = activation1+bias1
        if y1*activation1 <= 0:
            for word in count:
                if word in filteredfeatures:
                    weights1[word] = weights1[word]+(y1*count[word])
                    u1[word] = u1[word]+(y1*c*count[word])
            bias1 = bias1+y1
            beta1 += y1*c

        activation2 = activation2+bias2
        if y2*activation2 <= 0:
            for word in count:
                if word in filteredfeatures:
                    weights2[word] = weights2[word]+(y2*count[word])
                    u2[word] = u2[word]+(y2*c*count[word])
            bias2 = bias2+y2
            beta2 += y2*c
        c += 1
    k += 1


# print(weights)
with open("vanillamodel.txt", 'w') as f:
    f.write("weights1="+str(weights1)+"\n")
    f.write("bias1="+str(bias1)+"\n")
    f.write("weights2="+str(weights2)+"\n")
    f.write("bias2="+str(bias2)+"\n")
    f.write("filteredfeatures="+str(filteredfeatures)+"\n")

for i in weights1:
    weights1[i] = weights1[i]-(u1[i]/c)
    weights2[i] = weights2[i]-(u2[i]/c)
bias1 = bias1-(beta1/c)
bias2 = bias2-(beta2/c)

with open("averagedmodel.txt", 'w') as f:
    f.write("weights1="+str(weights1)+"\n")
    f.write("bias1="+str(bias1)+"\n")
    f.write("weights2="+str(weights2)+"\n")
    f.write("bias2="+str(bias2)+"\n")
    f.write("filteredfeatures="+str(filteredfeatures)+"\n")

