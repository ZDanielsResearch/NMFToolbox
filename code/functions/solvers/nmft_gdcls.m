function [W,H,FIters] = nmft_gdcls(A,W,H,params)
%%Paper:
% Text Mining using Non-Negative Matrix Factorization
% V. Pauca, F. Shahnaz, M. Berry, R. Plemmons
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
        error('Loss must be squared Euclidean for gradient descent-constrained least squares.');
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
    H = inv(W'*W + params.paramH.*eye(size(k,k)))*W'*A;
    H = H .* (H >= 0);
    W = W .* (A*H') ./ (W*H*H' + 1e-8);
    for i=1:1:k
        W(:,i) = W(:,i) ./ norm(W(:,i),2);
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