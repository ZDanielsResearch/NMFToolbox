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

params = [];
params.method = 'gdcls';
params.maxIters = 100;
params.initialization = 'nndsvd';
params.loss = 'sqeuclidean';
params.evalLoss = 'sqeuclidean';
params.stepType = 'steepest';
params.paramH = 0.5;
params.paramW = 0.5;
params.sparseParamH = 0.75;
params.sparseParamW = 0.75;
params.subIters = 1;
params.printIter = true;

[W,H,D] = nmft(data,numBasisElements,params);
