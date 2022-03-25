import csv, os
import math
from math import sqrt
from math import exp
from math import pi,log
import numpy as np
import pickle
from collections import defaultdict
import pathlib

class Gaussian_Bayes:

    def __init__(self, train_directory="processedData/trainingDataF.csv"):
        self.working_directory = os.getcwd()
        self.train_directory = self.working_directory + '/' + train_directory

        # File names with the stored model dictionary
        model_file = pathlib.Path(self.working_directory + '/' +  'model/pickled_model.data')
        # Check if pickled file exists, if true, loads it into memory, else calls train()
        if model_file.exists():
            self._trained_model = self.load(self.working_directory + '/' + 'model/pickled_model.data')
            #print(self._trained_model)
        else:
            self.train()
            self._trained_model = self.load(self.working_directory + '/' + 'model/pickled_model.data')

    def train(self):
        with open(self.train_directory, 'r') as f:
            self.train_dataset = list(csv.reader(f, delimiter=","))
        
        train_dataset = np.array(self.train_dataset[1:], dtype = 'float32')

        negative_train_dataset = train_dataset[np.where(train_dataset[:,4] == 0)]
        positive_train_dataset = train_dataset[np.where(train_dataset[:,4] == 1)]

        prior_positive = (len(positive_train_dataset) +1 )/(len(train_dataset)+1)
        prior_negative = (len(negative_train_dataset)+1)/(len(train_dataset)+1)

        positive_session_time = positive_train_dataset[:,0]
        positive_sex = positive_train_dataset[:,1]
        positive_age = positive_train_dataset[:,2]
        positive_task_type = positive_train_dataset[:,3]
        
        negative_session_time = negative_train_dataset[:,0]
        negative_sex = negative_train_dataset[:,1]
        negative_age = negative_train_dataset[:,2]
        negative_task_type = negative_train_dataset[:,3]
        
        # Mean and std for session time for both positive and negative result 
        negative_mean_st, negative_stdev_st = np.mean(negative_session_time), np.std(negative_session_time)        
        positive_mean_st, positive_stdev_st = np.mean(positive_session_time), np.std(positive_session_time)

        # Mean and std for age for both positive and negative result 
        negative_mean_age, negative_stdev_age = np.mean(negative_age), np.std(negative_age)
        positive_mean_age, positive_stdev_age = np.mean(positive_age), np.std(positive_age)

        # Positive and negative dictionaries for binary variables sex and task type
        # positive_sex_dict, negative_sex_dict, positive_tt_dict, negative_tt_dict

        ####################################################################################
        # Discrete variable's probabilites are calculated separately
        # Calculate probilities for binary variables sex (male or female) for positive and negative tasks
        count_pos_f, count_pos_m = 0,0
        for ps in positive_sex:
            if ps == 0:
                count_pos_f +=1
            elif ps == 1:
                count_pos_m +=1
        
        # Add one smoothing
        prob_male_given_positive = (count_pos_m + np.finfo(np.float64).eps) / (len(positive_sex) + np.finfo(np.float64).eps)
        prob_female_given_positive = (count_pos_f + np.finfo(np.float64).eps) / (len(positive_sex) + np.finfo(np.float64).eps)

        count_neg_f, count_neg_m = 0,0
        for ns in negative_sex:
            if ns == 0:
                count_neg_f +=1
            elif ns == 1:
                count_neg_m +=1
        
        # Add one smoothing
        prob_male_given_negative = (count_neg_m + np.finfo(np.float64).eps) / (len(negative_sex) + np.finfo(np.float64).eps)
        prob_female_given_negative = (count_neg_f+ np.finfo(np.float64).eps) / (len(negative_sex) + np.finfo(np.float64).eps)

        # Calculate probilities for binary variables task type (basic or full) for positive and negative tasks
        count_pos_basic, count_pos_full = 0,0
        for ptt in positive_task_type:
            if ptt == 0:
                count_pos_basic += 1
            elif ptt == 1:
                count_pos_full += 1
        
        # Add one smoothing
        prob_basic_task_given_positive = (count_pos_basic+ np.finfo(np.float64).eps) / (len(positive_task_type)+ np.finfo(np.float64).eps)
        prob_full_task_given_positive = (count_pos_full+ np.finfo(np.float64).eps) / (len(positive_task_type)+ np.finfo(np.float64).eps)

        count_neg_basic, count_neg_full = 0,0
        for ntt in negative_task_type:
            if ntt == 0:
                count_neg_basic +=1
            elif ntt == 1:
                count_neg_full +=1
        
        # Add one smoothing
        prob_basic_task_given_negative = (count_neg_basic + np.finfo(np.float64).eps) / (len(negative_sex)+ np.finfo(np.float64).eps)
        prob_full_task_given_negative = (count_neg_full+ np.finfo(np.float64).eps) / (len(negative_sex)+ np.finfo(np.float64).eps)
        
        ##################################################################################
        # Store information in dictionary and then pickle it for future usage

        training_dict= {
            "positive_sex_trainings" : {  1: prob_male_given_positive
                                        ,0: prob_female_given_positive
                                        },

            "negative_sex_trainings" : {  1: prob_male_given_negative
                                        ,0: prob_female_given_negative
                                        },

            "positive_task_trainings" : {   0:prob_basic_task_given_positive
                                                ,1:prob_full_task_given_positive
                                                },

            "negative_task_trainings" : {   0:prob_basic_task_given_negative
                                                ,1:prob_full_task_given_negative
                                                },

            "positive_age_trainings" : {  "mean": positive_mean_age
                                        ,"std": positive_stdev_age
                                        },

            "negative_age_trainings" : {  "mean": negative_mean_age
                                        ,"std": negative_stdev_age
                                        },

            "positive_session_time_trainings" : {
                                                "mean": positive_mean_st
                                                ,"std": positive_stdev_st
                                                },

            "negative_session_time_trainings" : {
                                                "mean": negative_mean_st
                                                ,"std": negative_stdev_st
                                                },
            "prior_probability": {
                                    "positive": 0.6
                                    ,"negative": 0.4
                    }             
        }

        self.save(training_dict,self.working_directory + '/' + 'model/pickled_model.data')
        ####################################################################################

    def classify(self, sex, age, session_time, task_type):
        # Since, we had converted standard data into z scores, we need to convert the new data into corresponding z score 
        # Training z mean and standard deviation is saved in processedData/zscoreMeansAndStdsF.csv

        with open(self.working_directory + '/' + 'processedData/zscoreMeansAndStdsF.csv', 'r') as f:
            z_mean_std_training = list(csv.reader(f, delimiter=","))
        
        z_mean_age, z_std_age = z_mean_std_training[1][0],z_mean_std_training[1][1]
        z_mean_st, z_std_st = z_mean_std_training[3][0],z_mean_std_training[3][1]

        # z_session_time = self.convert_z_value(session_time, z_mean_st, z_std_st)
        # z_age = self.convert_z_value(age, z_mean_age, z_std_age)
        
        z_session_time = session_time
        z_age = age

        prob_male_given_positive, prob_female_given_positive = self._trained_model['positive_sex_trainings'][1], self._trained_model['positive_sex_trainings'][0]
        prob_male_given_negative, prob_female_given_negative = self._trained_model['negative_sex_trainings'][1], self._trained_model['negative_sex_trainings'][0]
        
        prob_basic_task_given_positive, prob_full_task_given_positive = self._trained_model['positive_task_trainings'][0], self._trained_model['positive_task_trainings'][1]
        prob_basic_task_given_negative, prob_full_task_given_negative = self._trained_model['negative_task_trainings'][0], self._trained_model['negative_task_trainings'][1]

        positive_age_mean, positive_age_std = self._trained_model['positive_age_trainings']['mean'], self._trained_model['positive_age_trainings']['std']
        negative_age_mean, negative_age_std = self._trained_model['negative_age_trainings']['mean'], self._trained_model['negative_age_trainings']['std']

        positive_st_mean, positive_st_std = self._trained_model['positive_session_time_trainings']['mean'], self._trained_model['positive_session_time_trainings']['std']
        negative_st_mean, negative_st_std = self._trained_model['negative_session_time_trainings']['mean'], self._trained_model['negative_session_time_trainings']['std']

        prior_positive = self._trained_model['prior_probability']['positive']
        prior_negative = self._trained_model['prior_probability']['negative']

        prob_positive = np.log(prior_positive)
        prob_negative = np.log(prior_negative)
        
        # Calculate gaussian probability for continuous data
        prob_age_positive = self.univariate_gaussian_probability(z_age, positive_age_mean, positive_age_std)
        prob_age_negative = self.univariate_gaussian_probability(z_age, negative_age_mean, negative_age_std)

        prob_st_positive = self.univariate_gaussian_probability(z_session_time, positive_st_mean, positive_st_std)
        prob_st_negative = self.univariate_gaussian_probability(z_session_time, negative_st_mean, negative_st_std)

        
        prob_positive = np.log(prior_positive) + np.log(prob_age_positive) + np.log(prob_st_positive)
        prob_negative = np.log(prior_negative) + np.log(prob_age_negative) + np.log(prob_st_negative)

        if sex == 0 and task_type == 0: # If male and task type is basic
            prob_positive += (np.log(prob_male_given_positive + 1) + np.log(prob_basic_task_given_positive + 1))
            prob_negative += (np.log(prob_male_given_negative + 1) + np.log(prob_basic_task_given_negative + 1))

        elif sex ==0 and task_type == 1: #If male and task type is full
            prob_positive += (np.log(prob_male_given_positive + 1) + np.log(prob_full_task_given_positive + 1))
            prob_negative += (np.log(prob_male_given_negative + 1) + np.log(prob_full_task_given_negative + 1 ))
        
        elif sex ==1 and task_type == 0: #If female and task type is basic
            prob_positive += (np.log(prob_female_given_positive + 1) + np.log(prob_basic_task_given_positive + 1))
            prob_negative += (np.log(prob_female_given_negative + 1) + np.log(prob_basic_task_given_negative + 1))

        elif sex ==1 and task_type == 1: #If female and task type is full
            prob_positive += (np.log(prob_female_given_positive + 1) + np.log(prob_full_task_given_positive + 1))
            prob_negative += (np.log(prob_female_given_negative + 1) + np.log(prob_full_task_given_negative + 1))
        
        else:
            print("Invalid features entered.")
            exit(1)
        # Classification
        if prob_positive > prob_negative:
            #print("Task successful")
            return 1
        else:
            #print("Task unsuccessful")
            return 0

    def convert_z_value(self,val, mean, std):
        ''' Convert standard value into z value '''

        val, mean, std = float(val), float(mean), float(std)
        return (val - mean)/std


    # def univariate_gaussian_probability(self,x, mean, stdev):
    #     ''' Calculates univariate guassian probability based on mean and standard deviations passed'''

    #     exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    #     return (1 / (sqrt(2 * pi) * stdev)) * exponent

    def univariate_gaussian_probability(self, x, mean, stDeviation):
        sigma = stDeviation
        mew = mean

        probability = 1/(sigma*math.sqrt(2*math.pi)) * math.exp(- ((x-mew)**2)/(2*sigma**2))

        return probability

    def loadFile(self, sFilename):
        '''Given a file name, return the contents of the file as a string.'''

        f = open(sFilename, "r", encoding = 'UTF-8')
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


#############################################################################################
if __name__ =="__main__":
    c = Gaussian_Bayes()
    # c.classify(0,150,200,0)
    # c.classify(0,0.41339585,-0.38196287,1)
    c.classify(1,0.49181464,0.33519632,1)