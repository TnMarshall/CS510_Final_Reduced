import csv, os
from datetime import datetime
import numpy as np
from gaussian_naive_bayes import Gaussian_Bayes
import time
from datetime import datetime


start = time.time()
now = datetime.now()

start_time = now.strftime("%H:%M:%S")
print("Start Time =", start_time)

c = Gaussian_Bayes()

cwd = os.getcwd()
validationFile = cwd + "/processedData/validationDataF.csv"


with open(validationFile, 'r') as f:
    validationData = np.array(list(csv.reader(f, delimiter=","))[1:], dtype='float32')

features = validationData[:,0:4]
correctResult = validationData[:,4]

totalCount = 0
accurateCount = 0
TP = 0
TN = 0
FP = 0
FN = 0
for i in range(0,len(correctResult)):
    totalCount += 1
    actual = int(correctResult[i])
    predict = int(c.classify(features[i,1], features[i,2], features[i,0], features[i,3]))

    if (actual == 1) and (predict == 1):
        TP += 1
        accurateCount += 1
    elif (actual == 0) and (predict == 0):
        TN += 1
        accurateCount += 1
    elif (actual == 0) and (predict == 1):
        FP += 1
    elif (actual == 1) and (predict == 0):
        FN += 1

# DATA PROCESSING

accuracy = accurateCount / totalCount

precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2*(precision * recall) / (precision + recall)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F Measure: ", F1)


end = time.time()

# total time taken
print(f"Runtime of the program is {end - start} seconds")
