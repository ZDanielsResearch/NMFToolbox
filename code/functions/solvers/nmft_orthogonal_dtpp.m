function [W,H,FIters] = nmft_orthogonal_dtpp(A,W,H,params)
%%Paper:
% Orthogonal nonnegative matrix tri-factorizations for clustering
% C. Ding, X. T. Li, W. Peng, and H. Park
%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
% params.maxIters: Maximum number of iterations to perform
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

[n,k] = size(W);
[~,m] = size(H);

if ~isempty(params.loss)
    params.loss = lower(params.loss);
    if ~strcmp(params.loss,'sqeuclidean')
        error('Loss must be squared Euclidean for Choi orthogonal nonnegative matrix factorization.');
    end
end

for iterationNumber = 1:1:params.maxIters;
    H = H.*((W'*A)./(W'*W*H + 1e-8));
    W = W.*((A*H')./(W*W'*A*H'  + 1e-8)).^(0.5);
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