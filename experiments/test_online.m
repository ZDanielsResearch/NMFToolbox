setup;

k = 15;

[data,trueW,trueH] = generate_data_matrix(200000,500,k,0.1,1,100);
data = data';

FItersAll = zeros(51,5);

params = [];
params.method = 'online';
params.maxIters = 50;
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
params.orthogonalConstraint = 'w';
params.sample = 0.1;

[W,H,D,F,FIters] = nmft(data,k,params);
FItersAll(:,1) = FIters';

params = [];
params.method = 'online';
params.maxIters = 50;
params.initialization = 'nndsvd';
params.loss = 'sqeuclidean';
params.evalLoss = 'sqeuclidean';
params.stepType = 'newton';
params.paramH = 0.5;
params.paramW = 0.5;
params.sparseParamH = 0.75;
params.sparseParamW = 0.75;
params.subIters = 1;
params.printIter = true;
params.orthogonalConstraint = 'w';
params.sample = 0.1;

[W,H,D,F,FIters] = nmft(data,k,params);
FItersAll(:,2) = FIters';

params = [];
params.method = 'online';
params.maxIters = 50;
params.initialization = 'nndsvd';
params.loss = 'sqeuclidean';
params.evalLoss = 'sqeuclidean';
params.stepType = 'mixed';
params.paramH = 0.5;
params.paramW = 0.5;
params.sparseParamH = 0.75;
params.sparseParamW = 0.75;
params.subIters = 1;
params.printIter = true;
params.orthogonalConstraint = 'w';
params.sample = 0.1;

[W,H,D,F,FIters] = nmft(data,k,params);
FItersAll(:,3) = FIters';

params = [];
params.method = 'projgrad';
params.maxIters = 50;
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
params.orthogonalConstraint = 'w';
params.sample = 0.1;

[W,H,D,F,FIters] = nmft(data,k,params);
FItersAll(:,4) = FIters';

params = [];
params.method = 'als';
params.maxIters = 50;
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
params.orthogonalConstraint = 'w';
params.sample = 0.1;

[W,H,D,F,FIters] = nmft(data,k,params);
FItersAll(:,5) = FIters';

labels = {'online-pg','online-als','mixed','projgrad','als'};
colors = {'red','green','blue','black','magenta'};

scale = max(max(FItersAll)) ./ 8;

close all;
figure;
hold on;
for i = 1:1:2
    plot(FItersAll(2:51,i),'Color',colors{i})
    text(8,scale*i,labels{i},'Color',colors{i})
end
hold off;
