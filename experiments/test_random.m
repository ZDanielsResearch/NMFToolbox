setup;

k = 10;

data = generate_data_matrix(100,100,k,0.05,1,100);
data = data';

params = [];
params.method = 'linear';
params.maxIters = 50;
params.initialization = 'nndsvd';
params.loss = 'sqeuclidean';
params.evalLoss = 'sqeuclidean';
params.stepType = 'steepest';
params.paramH = 0.5;
params.paramW = 0.5;
params.sparseParamH = 0.75;
params.sparseParamW = 0.75;
params.subIters = 25;
params.printIter = true;
params.orthogonalConstraint = 'w';

[W,H,D,F,FIters] = nmft(data,k,params);

D

%%%%%%

params = [];
params.method = 'projgrad';
params.maxIters = 10;
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

[W,H,D,F,FIters] = nmft(data,k,params);

D

% FItersAll = zeros(10,7);
% FItersAll(:,1) = FIters';
% 
% params = [];
% params.method = 'projgrad';
% params.maxIters = 10;
% params.initialization = 'nndsvd';
% params.loss = 'sqeuclidean';
% params.evalLoss = 'sqeuclidean';
% params.stepType = 'steepest';
% params.paramH = 0.5;
% params.paramW = 0.5;
% params.sparseParamH = 0.75;
% params.sparseParamW = 0.75;
% params.subIters = 25;
% params.printIter = true;
% params.orthogonalConstraint = 'w';
% 
% [W,H,D,F,FIters] = nmft(data,k,params);
% FItersAll(:,2) = FIters';
% 
% params = [];
% params.method = 'projgrad2';
% params.maxIters = 10;
% params.initialization = 'nndsvd';
% params.loss = 'sqeuclidean';
% params.evalLoss = 'sqeuclidean';
% params.stepType = 'steepest';
% params.paramH = 0.5;
% params.paramW = 0.5;
% params.sparseParamH = 0.75;
% params.sparseParamW = 0.75;
% params.subIters = 25;
% params.printIter = true;
% params.orthogonalConstraint = 'w';
% 
% [W,H,D,F,FIters] = nmft(data,k,params);
% FItersAll(:,3) = FIters';
% 
% params = [];
% params.method = 'als';
% params.maxIters = 10;
% params.initialization = 'nndsvd';
% params.loss = 'sqeuclidean';
% params.evalLoss = 'sqeuclidean';
% params.stepType = 'newton';
% params.paramH = 0.5;
% params.paramW = 0.5;
% params.sparseParamH = 0.75;
% params.sparseParamW = 0.75;
% params.subIters = 1;
% params.printIter = true;
% params.orthogonalConstraint = 'w';
% 
% [W,H,D,F,FIters] = nmft(data,k,params);
% FItersAll(:,4) = FIters';
% 
% params = [];
% params.method = 'projgrad';
% params.maxIters = 10;
% params.initialization = 'nndsvd';
% params.loss = 'sqeuclidean';
% params.evalLoss = 'sqeuclidean';
% params.stepType = 'bfgs';
% params.paramH = 0.5;
% params.paramW = 0.5;
% params.sparseParamH = 0.75;
% params.sparseParamW = 0.75;
% params.subIters = 25;
% params.printIter = true;
% params.orthogonalConstraint = 'w';
% 
% [W,H,D,F,FIters] = nmft(data,k,params);
% FItersAll(:,5) = FIters';
% 
% params = [];
% params.method = 'mult';
% params.maxIters = 10;
% params.initialization = 'nndsvd';
% params.loss = 'sqeuclidean';
% params.evalLoss = 'sqeuclidean';
% params.stepType = 'steepest';
% params.paramH = 0.5;
% params.paramW = 0.5;
% params.sparseParamH = 0.75;
% params.sparseParamW = 0.75;
% params.subIters = 1;
% params.printIter = true;
% params.orthogonalConstraint = 'w';
% 
% [W,H,D,F,FIters] = nmft(data,k,params);
% FItersAll(:,6) = FIters';
% 
% params = [];
% params.method = 'augmented';
% params.maxIters = 10;
% params.initialization = 'nndsvd';
% params.loss = 'sqeuclidean';
% params.evalLoss = 'sqeuclidean';
% params.stepType = 'steepest';
% params.paramH = 0.5;
% params.paramW = 0.5;
% params.sparseParamH = 0.75;
% params.sparseParamW = 0.75;
% params.subIters = 25;
% params.printIter = true;
% params.orthogonalConstraint = 'w';
% 
% [W,H,D,F,FIters] = nmft(data,k,params);
% FItersAll(:,7) = FIters';
% 
% labels = {'steepestProj1','steepestProjN','full','als','bfgs','mult','augmented'};
% colors = {'red','green','blue','black','cyan','yellow','magenta'};
% 
% scale = max(max(FItersAll)) ./ 8;
% 
% close all;
% figure;
% hold on;
% for i = 1:1:7
%     plot(FItersAll(:,i),'Color',colors{i})
%     text(8,scale*i,labels{i},'Color',colors{i})
% end
% hold off;
