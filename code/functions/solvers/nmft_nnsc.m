function [W,H,FIters] = nmft_gdcls(A,W,H,params)
%%Paper:
% Non-Negative Sparse Coding
% P. Hoyer
%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
% params.maxIters: Maximum number of iterations to perform
% params.paramH: mixing coefficient associated with H
%Outputs
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m
% FIters: Sequence of function values

FIters = [];

if ~isempty(params.loss)
    params.loss = lower(params.loss);
    if ~strcmp(params.loss,'sqeuclidean')
        error('Loss must be squared Euclidean for nonnegative sparse coding.');
    end
end

if isempty(params.paramH)
    params.paramH = 0;
    disp('Warning: No params.paramH specified: Using params.paramH = 0.');
end

[n,k] = size(W);
[~,m] = size(H);

for i=1:1:k
    W(:,i) = W(:,i) ./ norm(W(:,i),2);
end

for iterationNumber=1:1:params.maxIters
    [~,~,dW,~,alphaW,~,~] = sqeuclidean_loss(A,W,H,[0 1],[0 0]);
    W = W - alphaW .* dW;
    W = W .* (W > 0);
    for i=1:1:k
        W(:,i) = W(:,i) ./ norm(W(:,i),2);
    end
    H = H .* (W'*A) ./ (W'*W*H + params.paramH);
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