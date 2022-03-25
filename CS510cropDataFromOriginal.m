clear; close all; clc
%% Separate out Columns 2,3,6,7,31, and 32

data = readtable('ibl_session_mice_data.csv');

dataToKeep = data(:,[2,3,6,7,31,32]);

writetable(dataToKeep, 'final510IBLdataReduced.csv')