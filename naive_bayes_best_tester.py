from bayesbest import Bayes_Classifier
import sys
import os
import numpy as np



def tester(trainDirectory = "./movie_reviews/", validationDirectory = "./movie_reviews/"):
    #initialize confusion matrix
    confusionMatrix = np.zeros((3,3), dtype=int)

    confusionMatrixLetters = np.array([['bb', 'bn', 'bg'],['nb', 'nn', 'ng'],['gb', 'gn', 'gg']])

    #initialize classifier
    bc = Bayes_Classifier(trainDirectory)

    # collect data to test the classifier on
    for fFileObj in os.walk(validationDirectory):
        IFileList = fFileObj[2]
        break
      
    numFiles = len(IFileList)
    accurateCount = 0
    totalCount = 0
    
    totalBad = 0
    totalGood = 0
    totalNeutral = 0

    # run each word through classify and check performance
    for i in range(0,numFiles): #10):#
        totalCount += 1
        indRate = IFileList[i].find('-') + 1
        rating = int(IFileList[i][indRate])
        fileContents = bc.loadFile(validationDirectory + IFileList[i])
        
        prediction = bc.classify(fileContents)
#        print("rating: ", rating)
#        print("prediction: ", prediction, "\n")
        if rating == prediction:
            accurateCount += 1
        elif ((rating == 2) or (rating == 3) or (rating == 4)) and prediction == 3:
            accurateCount += 1
        
        # count ratings
        if rating == 1:
            totalBad += 1
        elif rating == 5:
            totalGood += 1
        else:
            totalNeutral += 1


        # build confusion matrix
        if rating == 1 and prediction == 1:
            confusionMatrix[0,0] += 1
        elif rating == 1 and prediction == 3:
            confusionMatrix[0,1] += 1
        elif rating == 1 and prediction == 5:
            confusionMatrix[0,2] += 1
        
        elif ((rating == 2) or (rating == 3) or (rating == 4)) and prediction == 1:
            confusionMatrix[1,0] += 1
        elif ((rating == 2) or (rating == 3) or (rating == 4)) and prediction == 3:
            confusionMatrix[1,1] += 1
        elif ((rating == 2) or (rating == 3) or (rating == 4)) and prediction == 5:
            confusionMatrix[1,2] += 1

        elif rating == 5 and prediction == 1:
            confusionMatrix[2,0] += 1
        elif rating == 5 and prediction == 3:
            confusionMatrix[2,1] += 1
        elif rating == 5 and prediction == 5:
            confusionMatrix[2,2] += 1

    #calculate and output results
    print("Accuracy: ", (accurateCount/totalCount)*100)
    print(confusionMatrix)
    print("\nFor matrix, first letter = actual, second letter = predicted:")
    print(confusionMatrixLetters)
    print("\nTotal Actual Negative Reviews: ",totalBad)
    print("Total Actual Neutral Reviews: ", totalNeutral)
    print("Total Actual Positive Reviews: ", totalGood)

    badRecall = confusionMatrix[0,0] / (confusionMatrix[0,0]+confusionMatrix[0,1]+confusionMatrix[0,2])
    badPrecision = confusionMatrix[0,0] / (confusionMatrix[0,0]+confusionMatrix[1,0]+confusionMatrix[2,0])

    goodRecall = confusionMatrix[2,2] / (confusionMatrix[2,0]+confusionMatrix[2,1]+confusionMatrix[2,2])
    goodPrecision = confusionMatrix[2,2] / (confusionMatrix[0,2]+confusionMatrix[1,2]+confusionMatrix[2,2])

    if not (totalNeutral == 0):
        neutralRecall = confusionMatrix[1,1] / (confusionMatrix[1,0]+confusionMatrix[1,1]+confusionMatrix[1,2])
        neutralPrecision = confusionMatrix[1,1] / (confusionMatrix[0,1]+confusionMatrix[1,1]+confusionMatrix[2,1])
        overallPrecision = (goodPrecision + neutralPrecision + badPrecision)/3
        overallRecall = (goodRecall + neutralRecall + badRecall)/3
        overallFmeasure = (2*overallPrecision*overallRecall)/(overallPrecision+overallRecall)
    else:
        neutralPrecision = "NA"
        neutralRecall = "NA"
        overallPrecision = (goodPrecision + badPrecision)/2
        overallRecall = (goodRecall + badRecall)/2
        overallFmeasure = (2*overallPrecision*overallRecall)/(overallPrecision+overallRecall)
        

    # print("\nPositive Precision: ", goodPrecision)
    # print("Positive Recall", goodRecall)

    # print("Neutral Precision: ", neutralPrecision)
    # print("Neutral Recall: ", neutralRecall)

    # print("Negative Precision: ", badPrecision)
    # print("Negative Recall: ", badRecall)

    print("\nOverall Precision: ", overallPrecision)
    print("Overall Recall: ", overallRecall)
    print("Overall Fmeasure: ", overallFmeasure)


if __name__ == "__main__":
    # tester()
    # print("\n///////////////////////////////////////\nTraining Results:")
    # tester("./trainingDataMovies/", "./trainingDataMovies/")
    # print("\nValidation Results:")
    # tester("./trainingDataMovies/", "./validationDataMovies/")

    print("\n//////////////////BEST/////////////////////\nTraining Results:")
    tester("./trainingData/", "./trainingData/")
    print("\nValidation Results:")
    tester("./trainingData/", "./validationData/")
    
