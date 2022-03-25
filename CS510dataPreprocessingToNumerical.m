clear; close all; clc

%% read in data 
tic
data = readtable('final510IBLdataReduced.csv'); %final510IBLdataReduced.csv
toc
tic
subjectIDs = data(:,1);
SessionStartTimes = data(:,2);
sex = data(:,3);
birthDate = data(:,4);
taskType = data(:,5);
trialResult = data(:,6);
toc
pause(1)
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
pause(1)

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
pause(1)

% birthData %
tic
birthdateNumerical = zeros(size(birthDate));

for i = 1:size(birthDate,1)
   birthdateNumerical(i) = datenum(birthDate{i,'subject_birth_date'}); 
end
toc
% birthData %
pause(1) 

% session start time %
tic
sessionStartNumerical = zeros(size(SessionStartTimes));

for i = 1:size(sessionStartNumerical,1)
    sessionStartNumerical(i) = hour(SessionStartTimes{i,'session_start_time'}) * 60 + minute(SessionStartTimes{i,'session_start_time'});
end
toc
% session start time %


% trialResult %

trialResultNumerical = table2array(trialResult);

% trialResult %

numericalData = [sessionStartNumerical, sexNumerical, birthdateNumerical, taskTypeNumerical, trialResultNumerical];





%% Randomize Order


% Randomize data order %

shapeData = size(numericalData);
numCols = shapeData(2);
numRows = shapeData(1);
rng(0);
indicesRandomized = randperm(numRows);
randomizedData = zeros(size(numericalData));
randomizedSubjectIDs = subjectIDs;
for i=1:numRows
    randomizedData(indicesRandomized(i),:) = numericalData(i,:);
    randomizedSubjectIDs(indicesRandomized(i),'subject_uuid') = subjectIDs(i,'subject_uuid');
end

% Randomize data order %




% Seperate into training, validaiton, and test %

divider1 = ceil(numRows * 0.75);
divider2 = divider1 + ceil(numRows * 0.125);

trainingX = randomizedData(1:divider1, 1:4);
trainingClass = randomizedData(1:divider1, 5);
trainingSubjectID = randomizedSubjectIDs(1:divider1,'subject_uuid');

validationX = randomizedData((divider1+1):divider2, 1:4);
validationClass = randomizedData((divider1+1):divider2, 5);
validationSubjectID = randomizedSubjectIDs((divider1+1):divider2,'subject_uuid');

testX = randomizedData((divider2+1):end, 1:4);
testClass = randomizedData((divider2+1):end, 5);
testSubjectID = randomizedSubjectIDs((divider2+1):end,'subject_uuid');

% Seperate into training, validaiton, and test %



%% Zscore data %

zscoredTrainingX = zeros(size(trainingX));
zscoredValidationX = zeros(size(validationX));
zscoredTestX = zeros(size(testX));
means = zeros(numCols-1,1);
stds = zeros(numCols-1,1);

for index=1:(numCols-1) % -1 because the first column of data is the target data
    
    meanT = mean(trainingX(:,index));
    stdT = std(trainingX(:,index))+eps;
    means(index) = meanT;
    stds(index) = stdT;
    zscoredTrainingX(:,index) = (trainingX(:,index)-meanT)/stdT;
    
    zscoredValidationX(:,index) = (validationX(:,index)-meanT)/stdT;
    
    zscoredTestX(:,index) = (testX(:,index)-meanT)/stdT;
    
end

% Zscore data %


%% Recombine Data and export %

trainingOut = [trainingSubjectID, array2table(zscoredTrainingX), array2table(trainingClass)];
trainingOut.Properties.VariableNames = {'subject_uuid','session_start_time','sex','subject_birth_date','task_type','trial_result'};
writetable(trainingOut, 'processedData/trainingData.csv');

validationOut = [validationSubjectID, array2table(zscoredValidationX), array2table(validationClass)];
validationOut.Properties.VariableNames = {'subject_uuid','session_start_time','sex','subject_birth_date','task_type','trial_result'};
writetable(validationOut, 'processedData/validationData.csv');

testOut = [testSubjectID, array2table(zscoredTestX), array2table(testClass)];
testOut.Properties.VariableNames = {'subject_uuid','session_start_time','sex','subject_birth_date','task_type','trial_result'};
writetable(testOut, 'processedData/testData.csv');

zscore_means_and_stds = [array2table(means), array2table(stds)];
zscore_means_and_stds.Properties.VariableNames = {'means','stds'};
writetable(zscore_means_and_stds, 'processedData/zscoreMeansAndStds.csv');