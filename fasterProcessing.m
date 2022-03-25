clear; close all; clc

%% read in data 
tic
data = readtable('final510IBLdataReduced.csv'); %final510IBLdataReduced.csv
toc
tic
% subjectIDs = data(:,1);
SessionStartTimes = data(:,2);
sex = data(:,3);
birthDate = data(:,4);
taskType = data(:,5);
trialResult = data(:,6);
toc
% pause(1)
%% Process columns

%task type %
tic
taskTypeNumerical = zeros(size(taskType));

for i = 1:size(taskType,1)
    if taskType{i,'task_type'}{1} == "Full"
        taskTypeNumerical(i) = 1;
    else
        taskTypeNumerical(i) = 0;
    end
end
toc
%task type %
% pause(1)

% sex %
tic
sexNumerical = zeros(size(sex));

for i = 1:size(taskType,1)
    if sex{i,'sex'}{1} == 'F'
        sexNumerical(i) = 0;
    else
        sexNumerical(i) = 1;
    end
end
toc
% sex %
% pause(1)

% birthData %
tic
birthdateNumerical = zeros(size(birthDate));
ageNumerical = zeros(size(birthdateNumerical));

for i = 1:size(birthDate,1)
   birthdateNumerical(i) = datenum(birthDate{i,'subject_birth_date'});
   ageNumerical(i) = ceil(datenum(SessionStartTimes{i,'session_start_time'})-birthdateNumerical(i));
end
toc
% birthData %
% pause(1) 

% session start time %
tic
sessionStartNumerical = zeros(size(SessionStartTimes));

for i = 1:size(sessionStartNumerical,1)
    sessionStartNumerical(i) = round(hour(SessionStartTimes{i,'session_start_time'}) + minute(SessionStartTimes{i,'session_start_time'})/60);
end
toc
% session start time %


% trialResult %

trialResultNumerical = table2array(trialResult);

% trialResult %

numericalData = [sessionStartNumerical, sexNumerical, ageNumerical, taskTypeNumerical, trialResultNumerical];





%% Randomize Order


% Randomize data order %
tic
shapeData = size(numericalData);
numCols = shapeData(2);
numRows = shapeData(1);
rng(0);
indicesRandomized = randperm(numRows);
randomizedData = zeros(size(numericalData));
% randomizedSubjectIDs = subjectIDs;

% for i=1:numRows
%     randomizedData(indicesRandomized(i),:) = numericalData(i,:);
%     randomizedSubjectIDs(indicesRandomized(i),'subject_uuid') = subjectIDs(indicesRandomized(i),'subject_uuid');
% end
originalInds = [1:1:length(indicesRandomized)];

randomizedData(indicesRandomized, :) = numericalData(originalInds,:);
% randomizedSubjectIDs(indicesRandomized, 'subject_uuid') = subjectIDs(originalInds, 'subject_uuid');
toc
% Randomize data order %




% Seperate into training, validaiton, and test %
tic
divider1 = ceil(numRows * 0.75);
divider2 = divider1 + ceil(numRows * 0.125);

trainingX = randomizedData(1:divider1, 1:4);
trainingClass = randomizedData(1:divider1, 5);
% trainingSubjectID = randomizedSubjectIDs(1:divider1,'subject_uuid');

validationX = randomizedData((divider1+1):divider2, 1:4);
validationClass = randomizedData((divider1+1):divider2, 5);
% validationSubjectID = randomizedSubjectIDs((divider1+1):divider2,'subject_uuid');

testX = randomizedData((divider2+1):end, 1:4);
testClass = randomizedData((divider2+1):end, 5);
% testSubjectID = randomizedSubjectIDs((divider2+1):end,'subject_uuid');
toc
% Seperate into training, validaiton, and test %



%% Zscore data %
tic
zscoredTrainingX = zeros(size(trainingX));
zscoredValidationX = zeros(size(validationX));
zscoredTestX = zeros(size(testX));
means = zeros(numCols-1,1);
stds = zeros(numCols-1,1);

for index=[1,3]% session start and age, the others are binary and the last is success 1:(numCols-1) % -1 because the last column of data is the target data
    
    meanT = mean(trainingX(:,index));
    stdT = std(trainingX(:,index))+eps;
    means(index) = meanT;
    stds(index) = stdT;
    zscoredTrainingX(:,index) = (trainingX(:,index)-meanT)/stdT;
    
    zscoredValidationX(:,index) = (validationX(:,index)-meanT)/stdT;
    
    zscoredTestX(:,index) = (testX(:,index)-meanT)/stdT;
    
end
toc
% Zscore data %

zscoredTrainingX(:,2) = trainingX(:,2);
zscoredTrainingX(:,4) = trainingX(:,4);

zscoredValidationX(:,2) = validationX(:,2);
zscoredValidationX(:,4) = validationX(:,4);

zscoredTestX(:,2) = testX(:,2);
zscoredTestX(:,4) = testX(:,4);

%% Recombine Data and export %
tic
trainingOut = [array2table(zscoredTrainingX), array2table(trainingClass)];
trainingOut.Properties.VariableNames = {'session_start_time','sex','subject_age_at_testing','task_type','trial_result'};
writetable(trainingOut, 'processedData/trainingDataF.csv');

validationOut = [array2table(zscoredValidationX), array2table(validationClass)];
validationOut.Properties.VariableNames = {'session_start_time','sex','subject_age_at_testing','task_type','trial_result'};
writetable(validationOut, 'processedData/validationDataF.csv');

testOut = [array2table(zscoredTestX), array2table(testClass)];
testOut.Properties.VariableNames = {'session_start_time','sex','subject_age_at_testing','task_type','trial_result'};
writetable(testOut, 'processedData/testDataF.csv');

zscore_means_and_stds = [array2table(means), array2table(stds)];
zscore_means_and_stds.Properties.VariableNames = {'means','stds'};
writetable(zscore_means_and_stds, 'processedData/zscoreMeansAndStdsF.csv');
toc