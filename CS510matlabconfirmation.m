clear; close all; clc

% read in data %

data = readtable('ibl_session_mice_data.csv');
data = data{:,:}; %convert table to array
datasize = size(data);
data = data(:, 1:datasize(2) );

% read in data %


% Randomize data order %

shapeData = size(data);
numCols = shapeData(2);
numRows = shapeData(1);
rng(0);
indicesRandomized = randperm(numRows);
randomizedData = zeros(size(data));
for i=1:numRows
    randomizedData(indicesRandomized(i),:) = data(i,:);
end

% Randomize data order %



% Seperate into X and Y %

Xall = randomizedData(:, 1:end-1 );
Yall = randomizedData(:,end);

% Seperate into X and Y %


% Seperate into training and validaiton %

divider = ceil(numRows * 2/3);

trainingX = Xall(1:divider, :);
trainingClass = Yall(1:divider, :);

validationX = Xall((divider+1):end, :);
validationClass = Yall((divider+1):end, :);

% Seperate into training and validaiton %


% Zscore data %

zscoredTrainingX = zeros(size(trainingX));
zscoredValidationX = zeros(size(validationX));
means = zeros(numCols-1);
stds = zeros(numCols-1);

for index=1:(numCols-1) % -1 because the first column of data is the target data
    
    meanT = mean(trainingX(:,index));
    stdT = std(trainingX(:,index));
    means(index) = meanT;
    stds(index) = stdT;
    zscoredTrainingX(:,index) = (trainingX(:,index)-meanT)/stdT;
    
    zscoredValidationX(:,index) = (validationX(:,index)-meanT)/stdT;
    
end

% Zscore data %