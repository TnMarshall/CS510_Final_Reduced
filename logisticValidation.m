clear; close all; clc
%% Validation

tic
data = readtable('./processedData/validationDataF.csv');
toc

validationX = [ones(size(data{:,1},1),1),data{:,1:4}];
class = data{:,5};

tic
weightData = readtable('./backupofweights.txt','FileType','text');
toc

W = weightData{:,1};

yhat = 1./(1+exp(-validationX*W));

% % Apply all three models to validation data %
% Xvalid = [ones(size(zscoredValidationX,1),1),validationX];
% 
% yhat_1_2 = 1./(1+exp(-Xvalid*Weights_1_2));
% yhat_2_3 = 1./(1+exp(-Xvalid*Weights_2_3));
% yhat_3_1 = 1./(1+exp(-Xvalid*Weights_3_1));
% 
% is1 = yhat_1_2 + (1-yhat_3_1);
% is2 = (1-yhat_1_2) + yhat_2_3;
% is3 = yhat_3_1 + (1-yhat_2_3);
% 
% % Apply all three models to validation data %

% classify %

predict = zeros(size(yhat));
meanOfYhat = mean(yhat);
for i = 1:size(yhat,1)
    if yhat(i) > meanOfYhat
        predict(i) = 0;
    else
        predict(i) = 1;
    end
end

accurateCount = 0;
TP = 0;
FP = 0;
TN = 0;
FN = 0;
for i = 1:size(yhat,1)
    if (class(i) == 1) & (predict(i) == 1)
        accurateCount = accurateCount + 1;
        TP = TP + 1;
    elseif (class(i) == 0) & (predict(i) == 0)
        accurateCount = accurateCount + 1;
        TN = TN + 1;
    elseif (class(i) == 1) & (predict(i) == 0)
        FN = FN + 1;
    elseif (class(i) == 0) & (predict(i) == 1)
        FP = FP + 1;
    end
    
end

accuracy = accurateCount / i
% accuracy2 = (TP+TN) / (TP+FP+FN+TN)
confusion = [ TP, FP; FN, TN ]
confusionPerc = confusion / sum(sum(confusion))*100
precision = TP / (TP+FP)
recall = TP / (TP+FN)
Fmeasure = 2 * (precision*recall) / (precision+recall)