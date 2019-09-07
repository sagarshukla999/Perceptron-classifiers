import sys
import os
import re
import ast
import math
import glob
from collections import Counter

testing_data = sys.argv[2]

model_file = sys.argv[1]
output_file = "percepoutput.txt"
open(output_file, 'w').close()
items = []
with open(model_file, 'r') as i:
    line = i.readline()
    while line:
        items.append(line)
        line = i.readline()

weights1 = {}
bias1 = 0
weights2 = {}
bias2 = 0
for item in items:
    value = item.split("=")
    if value[0] == "weights1":
        weights1 = ast.literal_eval(value[1])
    elif value[0] == "bias1":
        bias1 = float(value[1])
    elif value[0] == "weights2":
        weights2 = ast.literal_eval(value[1])
    elif value[0] == "bias2":
        bias2 = float(value[1])
    elif value[0] == "filteredfeatures":
        filteredfeatures = ast.literal_eval(value[1])

    else:
        print("Error")


allpaths = glob.glob(os.path.join(testing_data, '*/*/*/*.txt'))


for paths in allpaths:
    with open(paths, 'r') as i:
        lines = i.read()
    lines = lines.lower()
    lines = re.sub(r"[^a-zA-Z]+", ' ', lines)
    arr = lines.split()
    count = Counter(arr)

    activation1 = 0
    activation2 = 0
    for i in count:
        if i in filteredfeatures:
            activation1 += weights1[i]*count[i]
            activation2 += weights2[i]*count[i]
    activation1 += bias1
    activation2 += bias2

    ans = ""
    if activation1 > 0:
        if activation2 > 0:
            ans = "truthful positive "+paths
        else:
            ans = "deceptive positive "+paths
    else:
        if activation2 > 0:
            ans = "truthful negative "+paths
        else:
            ans = "deceptive negative "+paths

    with open(output_file, 'a') as f:
        f.write(ans+"\n")
