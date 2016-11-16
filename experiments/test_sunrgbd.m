setup;

load('data/sunrgbd_information.mat');

[numInstances,numObjects] = size(objectList);

%Parameters
numBasisElements = 50;
maxIters = 200;

%Print and plot statistics
disp(['The number of instances is ' num2str(numInstances)]);
disp(['The number of objects is ' num2str(numObjects)]);

close all;
figure;
bar(sum(objectList));
title('Distribution of Object Frequencies');
xlabel('Object');
ylabel('Frequency');

%Perform IDF weighing
objectList = objectList + 1e-10;
encoderWeights = log(1 + (numInstances ./ sum(objectList)));
data = objectList .* repmat(encoderWeights,[numInstances,1]);
data = data';

clear objectList;

[W,H,D] = nmft(data,numBasisElements,'als',maxIters,'random');
