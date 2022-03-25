clear; close all; clc
%% Apply logisticBinaryClassifier

tic
data = readtable('./processedData/trainingDataF.csv'); %final510IBLdataReduced.csv
toc

tic
data = data{:,:};
features = data(:,1:4);
class = data(:,5);
toc

% Train the logistic binary classifier
tic
[TP, FP, TN, FN, Weights] = logisticBinaryClassifier(features, class, 0.5, 1, 10^(-4));
toc