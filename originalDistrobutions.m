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

%% Age at testing, distrobution of start times, task types, result

tic
numericalBirthDate = datenum(birthDate{:,'subject_birth_date'});
numericalAge = ceil(datenum(SessionStartTimes{:,'session_start_time'})-numericalBirthDate(:));
toc

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

tic
sessionStartNumerical = round(hour(SessionStartTimes{:,'session_start_time'}) + minute(SessionStartTimes{:,'session_start_time'})/60);
toc


%task Result %
tic

trialResultNumerical = trialResult{:,:};

toc
%task Result %


% mean and std of continuous vars %

meanAge = mean(numericalAge);
stdAge = std(numericalAge);

meanSessionStart = mean(sessionStartNumerical);
stdSessionStart = std(sessionStartNumerical);

% mean and std of continuous vars %


% binary data info %

numberTrialFull = sum(taskTypeNumerical);
numberTrialBasic = length(taskTypeNumerical)-numberTrialFull;

numberTrialSuccess = sum(trialResultNumerical);
numberTrialFailure = length(trialResultNumerical) - numberTrialSuccess;

% binary data info %


%% Plots

ageX = meanAge - 5*stdAge: 0.01: meanAge + 5*stdAge;
ageY = normpdf(ageX, meanAge, stdAge);

startX = meanSessionStart - 5*stdSessionStart : 0.01: meanSessionStart + 5*stdSessionStart;
startY = normpdf(startX, meanSessionStart, stdSessionStart);

figure(1)
plot(ageX,ageY)
xlabel("Age of Mouse")
title("Normal Distrobution of the Ages of the Mice at Testing")

figure(2)
plot(startX, startY)
title("Normal Distrobution of the Start Times of Testing")
xlabel("Session Start Time")

figure(3)
bar([numberTrialBasic, numberTrialFull])
title("Number of Basic and Full Trials")
names={'Basic'; 'Full'};
set(gca,'xticklabel',names)

figure(4)
bar([numberTrialSuccess, numberTrialFailure])
title("Number of Trial Successes and Failures")
names={'Success'; 'Failure'};
set(gca,'xticklabel',names)