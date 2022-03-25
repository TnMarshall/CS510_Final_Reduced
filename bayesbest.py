import math, os, pickle, re, sys, numpy as np
import statistics as st
from univariateGaussian import univariateGaussianFunc

class Bayes_Classifier:

    def checkClassMeanVar(self):
        trainDirectory = self.trainDirectory
        for fFileObj in os.walk(trainDirectory):
                IFileList = fFileObj[2]
                break
            
        numFiles = len(IFileList)

        totalBad = 0
        totalGood = 0
        totalNeutral = 0
        total = 0

        goodLengths = []
        badLengths = []
        neutralLengths = []
        

        for i in range(0,numFiles):
            total += 1

            indRate = IFileList[i].find('-') + 1
            rating = int(IFileList[i][indRate])
            fileContents = self.loadFile(trainDirectory + IFileList[i])
            words = self.tokenize(fileContents)
            numWords = len(words)

            # count ratings
            if rating == 1:
                totalBad += 1
                badLengths.append(numWords)
            elif rating == 5:
                totalGood += 1
                goodLengths.append(numWords)
            else:
                totalNeutral += 1
                neutralLengths.append(numWords)

        if len(goodLengths) > 1:
            goodMean = st.mean(goodLengths)
            goodStd = st.stdev(goodLengths)
        else:
            goodMean = 0
            goodStd = 0
        
        if len(badLengths) > 1:
            badMean = st.mean(badLengths)
            badStd = st.stdev(badLengths)
        else:
            badMean = 0
            badStd = 0

        if len(neutralLengths) > 1:
            neutralStd = st.stdev(neutralLengths)
            neutralMean = st.mean(neutralLengths)
        else:
            neutralStd = 0
            neutralMean = 0

        self.goodMean = goodMean
        self.neutralMean = neutralMean
        self.badMean = badMean

        self.goodStd = goodStd
        self.neutralStd = neutralStd
        self.badStd = badStd

        # print(goodMean)
        # print(neutralMean)
        # print(badMean)

        # print(goodStd)
        # print(neutralStd)
        # print(badStd)

    def __init__(self, trainDirectory = "./movie_reviews/"):
        self.trainDirectory = trainDirectory
        '''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
        cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
        the system will proceed through training.  After running this method, the classifier 
        is ready to classify input text.'''
        #load dictionaries if they are present. If not, train dictionaries from training dictionary
        try:
            self.goodDict = self.load("goodDictBest.pkl")
            self.badDict = self.load("badDictBest.pkl")
        except OSError as err:
            self.train()

        # Determine how many words are in each category.
        self.goodCount = 0
        self.badCount = 0

        for i in self.goodDict:
            self.goodCount += self.goodDict[i]
        
        for i in self.badDict:
            self.badCount += self.badDict[i]

        # print(self.goodCount)
        # print(self.badCount)

        # Calculate means and stds for distrobution
        self.checkClassMeanVar()

        # Count the number of good, neutral, and bad ratings in the training data for
        # calculation of priors
        goodRate = 0
        neutralRate = 0
        badRate = 0

        for fFileObj in os.walk(self.trainDirectory):
            IFileList = fFileObj[2]
            break
      
        numFiles = len(IFileList)

        for i in range(0,numFiles):
            indRate = IFileList[i].find('-') + 1
            rating = int(IFileList[i][indRate])
            if rating == 1:
                badRate += 1
            elif rating == 5:
                goodRate += 1
            else:
                neutralRate += 1

        #priors based on training counts

        self.goodPrior = (goodRate / (goodRate + neutralRate + badRate))
        self.neutralPrior = neutralRate / (goodRate + neutralRate + badRate)
        self.badPrior = (badRate / (goodRate + neutralRate + badRate))

        # priors based on word counts
        # self.goodPrior = self.goodCount/(self.goodCount+self.badCount+self.neutralCount)
        # self.badPrior = self.badCount/(self.goodCount+self.badCount+self.neutralCount)
        # self.neutralPrior = self.neutralCount/(self.goodCount+self.badCount+self.neutralCount)

        self.totalWords = self.goodCount + self.badCount

        # print(self.goodPrior)
        # print(self.neutralPrior)
        # print(self.badPrior)

    

    def train(self):   
        '''Trains the Naive Bayes Sentiment Classifier.'''
        IFileList = []
        GoodDict = {}
        BadDict = {}
        allWords = {}

        for fFileObj in os.walk(self.trainDirectory):
            IFileList = fFileObj[2]
            break
      
        numFiles = len(IFileList)
      
        # retrieve all words in a the files and add one for each appearance to the proper dictionaries.
        for i in range(0,numFiles):
            indRate = IFileList[i].find('-') + 1
            rating = int(IFileList[i][indRate])
            if rating == 1:
                fileContents = self.loadFile(self.trainDirectory + IFileList[i])
                words = self.tokenize(fileContents)
                numWords = len(words)

                for j in range(0,numWords):
                    if not(words[j] in allWords):
                        allWords[words[j]] = 1

                    if words[j] in BadDict:
                        BadDict[words[j]] += 1
                    else:
                        BadDict[words[j]] = 1
                # print(BadDict)
                # print("Bad")
            elif rating == 5:
                fileContents = self.loadFile(self.trainDirectory + IFileList[i])
                words = self.tokenize(fileContents)
                numWords = len(words)

                for j in range(0,numWords):
                    if not(words[j] in allWords):
                        allWords[words[j]] = 1

                    if words[j] in GoodDict:
                        GoodDict[words[j]] += 1
                    else:
                        GoodDict[words[j]] = 1

                # print("Good")
            else:
                pass

        # add one smoothing
        alpha = 0
        for i in allWords:
            if i in BadDict:
                BadDict[i] += 1-alpha
            else:
                BadDict[i] = 1-alpha
            
            if i in GoodDict:
                GoodDict[i] += 1-alpha
            else:
                GoodDict[i] = 1-alpha

        # print(GoodDict, end="\n\n")
        # print(BadDict, end="\n\n")
        # print(NeutralDict, end="\n\n")

        # Store dictionaries for current use and for future use
        self.goodDict = GoodDict
        self.badDict = BadDict
        self.save(GoodDict, "goodDictBest.pkl")
        self.save(BadDict, "badDictBest.pkl")

    
    def classify(self, sText):
        '''Given a target string sText, this function returns the most likely document
        class to which the target string belongs. This function should return one of three
        strings: "positive", "negative" or "neutral".
        '''
        words = self.tokenize(sText)
        # working in log space, take log of priors
        positiveProb = math.log10(self.goodPrior)
        negativeProb = math.log10(self.badPrior)

        numWords = len(words)

#        print("Pos Prior log10: ", positiveProb)
#        print("Neg Prior log10: ", negativeProb)

        for i in words:
            # Retrieve counts of word in each class
            if i in self.goodDict:
                goodCount = self.goodDict[i]
            else:
                goodCount = 0
            if i in self.badDict:
                badCount = self.badDict[i]
            else:
                badCount = 0
            
            #count of a specific word divided by total number of words for a class
            likelihoodGood = math.log10( (goodCount+(np.finfo(np.float64).eps)) / (self.goodCount+(np.finfo(np.float64).eps)) )
            likelihoodBad = math.log10( (badCount+(np.finfo(np.float64).eps)) / (self.badCount+(np.finfo(np.float64).eps)) )

            #probability for the word overall
            wordProb = math.log10((goodCount+badCount)+(np.finfo(np.float64).eps)/ (self.goodCount + self.badCount))


            # sum of (likelihood given class) * prior of class / probability of the word appearing over all classes
            positiveProb += likelihoodGood# / wordProb
            negativeProb += likelihoodBad# / wordProb

#            print("Positive: ", positiveProb)
#            print("Negative: ", negativeProb)
            weight = numWords
            positiveProb += weight*math.log10(univariateGaussianFunc(numWords, self.goodMean, self.goodStd))
            negativeProb += weight*math.log10(univariateGaussianFunc(numWords, self.badMean, self.badStd))

            # neutral if positive and negative are close
            difPosNeg = abs(abs(positiveProb)-abs(negativeProb))
            # print("Diff: ", difPosNeg)
            #determine class based on probabilites. If the probabilities of good and bad are close, select neutral
            #use higher threshold, like 0.15 for data with neutral content.
            if difPosNeg < 0.3: #0.15
                return 3
            elif positiveProb > negativeProb:
                return 5
            else:
                return 1




    def loadFile(self, sFilename):
      '''Given a file name, return the contents of the file as a string.'''

      f = open(sFilename, "r", encoding="UTF-8")
      sTxt = f.read()
      f.close()
      return sTxt
   
    def save(self, dObj, sFilename):
      '''Given an object and a file name, write the object to the file using pickle.'''

      f = open(sFilename, "wb")
      p = pickle.Pickler(f)
      p.dump(dObj)
      f.close()
   
    def load(self, sFilename):
      '''Given a file name, load and return the object stored in the file.'''

      f = open(sFilename, "rb")
      u = pickle.Unpickler(f)
      dObj = u.load()
      f.close()
      return dObj

    def tokenize(self, sText): 
      '''Given a string of text sText, returns a list of the individual tokens that 
      occur in that string (in order).'''

      lTokens = []
      sToken = ""
      for c in sText:
         if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
            sToken += c
         else:
            if sToken != "":
               lTokens.append(sToken)
               sToken = ""
            if c.strip() != "":
               lTokens.append(str(c.strip()))
               
      if sToken != "":
         lTokens.append(sToken)

      return lTokens
