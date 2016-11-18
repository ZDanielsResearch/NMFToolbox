function [W,H,F] = nmft_convex(A,W,H,params)
%%Paper:
% Convex and Semi-Negative Matrix Factorizations
% C. Ding, T. Li, M. Jordan
%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
% params.maxIters: Maximum number of iterations to perform
%Outputs
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m

[n,k] = size(W);
[~,m] = size(H);

if ~isempty(params.loss)
    params.loss = lower(params.loss);
    if ~strcmp(params.loss,'sqeuclidean')
        error('Loss must be squared Euclidean for convex nonnegative matrix factorization.');
    end
end

H = H + 0.2.*ones(k,m);
F = H'*inv(H*H');
F = F .* (F > 0);
F = F + 0.2.*ones(m,k).*sum(sum(F))./nnz(F);

for iterationNumber = 1:1:params.maxIters;
    
    H = H .* (W'*A) ./ (W'*W*H + 1e-8);
    F = F .* ((A'*A*H') ./ (A'*W*H*H')).^(0.5);
    W = A*F;
    if params.printIter
        if strcmp(params.evalLoss,'sqeuclidean')
            [F1,~,~,~,~,~,~] = sqeuclidean_loss(A,W,H,[0 0],[0 0]);
        end
        if strcmp(params.evalLoss,'kldivergence')
            F1 = kl_loss(A,W,H);
        end
        disp(['Iteration #' num2str(iterationNumber) ', Function Value: ' num2str(F1)]);
    end
end

W = A*F;