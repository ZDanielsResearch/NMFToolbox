function [W,H,FIters] = nmft_hoyer(A,W,H,params)
%%Paper:
% Non-Negative Matrix Factorization with Sparseness Constraints
% P. Hoyer
%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
% params.maxIters: Maximum number of iterations to perform
% params.sparseParamH: parameter for Hoyer sparsity associated with H
% params.sparseParamW: parameter for Hoyer sparsity associated with W
%Outputs
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m
% FIters: Sequence of function values

FIters = [];
if params.printIter
    if strcmp(params.evalLoss,'sqeuclidean')
        [F,~,~,~,~,~,~] = sqeuclidean_loss(A,W,H,[0 0],[0 0]);
    end
    if strcmp(params.evalLoss,'kldivergence')
        F = kl_loss(A,W,H);
    end
    FIters = [FIters; F];
    disp(['Iteration #' num2str(0) ', Function Value: ' num2str(F)]);
end

if ~isempty(params.loss)
    params.loss = lower(params.loss);
    if ~strcmp(params.loss,'sqeuclidean')
        error('Loss must be squared Euclidean for nonnegative sparse coding.');
    end
end

if isempty(params.sparseParamH)
    params.sparseParamH = 0;
    disp('Warning: No params.sparseParamH specified: Using params.sparseParamH = 0.');
end

if isempty(params.sparseParamW)
    params.sparseParamW = 0;
    disp('Warning: No params.sparseParamW specified: Using params.sparseParamW = 0.');
end

[n,k] = size(W);
[~,m] = size(H);

W = W + rand(n,k).* 0.0001;
H = H + rand(k,m).* 0.0001;

k11 = sqrt(k)-(sqrt(k)-1)*params.sparseParamW;
k12 = sqrt(k)-(sqrt(k)-1)*params.sparseParamH;

for i = 1:1:k
    W(:,i) = W(:,i) ./ norm(W(:,i),2);
    W(:,i) = sparseness(W(:,i),k11,1,true);
end

for i = 1:1:m
    H(:,i) = H(:,i) ./ norm(H(:,i),2);
    H(:,i) = sparseness(H(:,i),k12,1,true);
end

for iterationNumber=1:1:params.maxIters
    [~,~,dW,~,alphaW,~,~] = sqeuclidean_loss(A,W,H,[0 1],[0 0]);
    W = W - alphaW .* dW;
    W = W .* (W > 0);
    for i = 1:1:k
        W(:,i) = W(:,i) ./ norm(W(:,i),2);
        W(:,i) = sparseness(W(:,i),k11,1,true);
    end
    
    [~,dH,~,alphaH,~,~,~] = sqeuclidean_loss(A,W,H,[1 0],[0 0]);
    H = H - alphaH .* dH;
    H = H .* (H > 0);
    for i = 1:1:m
        H(:,i) = H(:,i) ./ norm(H(:,i),2);
        H(:,i) = sparseness(H(:,i),k12,1,true);
    end
            
    if params.printIter
        F = 0;
        if strcmp(params.evalLoss,'sqeuclidean')
            [F,~,~,~,~,~,~] = sqeuclidean_loss(A,W,H,[0 0],[0 0]);
        end
        if strcmp(params.evalLoss,'kldivergence')
            F = kl_loss(A,W,H);
        end
        FIters = [FIters; F];
        disp(['Iteration #' num2str(iterationNumber) ', Function Value: ' num2str(F)]);
    end
end

end