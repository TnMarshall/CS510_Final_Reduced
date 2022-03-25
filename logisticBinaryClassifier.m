function [TP, FP, TN, FN, Weights] = logisticBinaryClassifier(features, classData, threshold, goalClassName, eta)
    %input data is expected to already be zscored and randomized

    lenData = length(classData);
    numCols = size(features,2);
    classDataNumeric = zeros(size(classData));
    
%     for iLocal = 1:lenData
%         if classData(iLocal) == goalClassName
%             classDataNumeric(iLocal) = 1;
%         else
%             classDataNumeric(iLocal) = 0;
%         end
%     end
    classData = classDataNumeric;
    
    % Prep for epochs %

    lenTraining = size(features,1);

    Y = classData;
    X = [ones(lenTraining,1), features];

    epochs = 1000000;
%     eta = 10^-4; 

    %starting weights
    rng(0)
    a = -(10^(-4));
    b = 10^(-4);
    W = (a + (b-a).*rand(numCols+1,1));
    %starting weights

    meanLogLikelihood = zeros(epochs,1);

    % Prep for epochs %
    
    
    % EPOCHS %

    for iLocal = 1:epochs

        yhat = 1./(1+exp(-X*W));

        djdw = (X'*(Y-yhat))/lenTraining;

        meanLogLikelihood(iLocal) = sum(Y .* log(yhat) + (1 - Y) .* log (1 - yhat)) / lenTraining;

        W = W + eta * djdw;

        if ((iLocal>1) && ( (abs(meanLogLikelihood(iLocal) - meanLogLikelihood(iLocal-1))) < (2^(-32))))
            break;
        end

    end

    % EPOCHS %
    
    
    % TP FP TN FN %
    Yhat = 1./(1+exp(-X*W));
    Ypredict = zeros(size(Yhat));
    
    for i = 1: lenTraining
        if Yhat(i) > threshold
            Ypredict(i) = 1;
        end
    end

    TP = 0;
    FP = 0;
    TN = 0;
    FN = 0;
    for i = 1:lenTraining
        if ((Ypredict(i) == 1) && (Y(i) == 1))
            TP = TP + 1;
        elseif ((Ypredict(i) == 1) && (Y(i) == 0))
            FP = FP + 1;    
        elseif ((Ypredict(i) == 0) && (Y(i) == 1))
            FN = FN + 1;
        elseif ((Ypredict(i) == 0) && (Y(i) == 0))
            TN = TN + 1;
        end
    end

    Weights = W;
end