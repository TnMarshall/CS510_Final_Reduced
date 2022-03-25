clear; close all; clc
data = readtable('final510IBLdataReduced.csv');

dataOut = data(1:10000,:);

writetable(dataOut, 'testSet10000.csv');