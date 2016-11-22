function [W,H,FIters] = nmft_orthogonal_choi(A,W,H,params)
%%Paper:
% Algorithms for Orthogonal Nonnegative Matrix Factorization
% S. Choi
%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
% params.maxIters: Maximum number of iterations to perform
% params.orthogonalConstraint: Enforce orthogonality in {'w','h'} for Choi algorithm
%Outputs
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m
% FIters: Sequence of function values

FIters = [];

[n,k] = size(W);
[~,m] = size(H);

if ~isempty(params.loss)
    params.loss = lower(params.loss);
    if ~strcmp(params.loss,'sqeuclidean')
        error('Loss must be squared Euclidean for Choi orthogonal nonnegative matrix factorization.');
    end
end

if isempty(params.orthogonalConstraint)
    params.orthogonalConstraint = 'w';
    disp('Warning: No orthogonal constraint specified: Enforcing (W^T)W = I.');
end

for i = 1:1:k
    W(:,i) = W(:,i) ./ norm(W(:,i),2);
end

for i = 1:1:k
    H(i,:) = H(i,:) ./ norm(H(i,:),2);
end

params.orthogonalConstraint = lower(params.orthogonalConstraint);
for iterationNumber = 1:1:params.maxIters;
    switch params.orthogonalConstraint
        case 'w'
            W = W.*((A*H')./(W*H*A'*W + 1e-8));
            for i = 1:1:k
                W(:,i) = W(:,i) ./ norm(W(:,i),2);
            end
            H = H.*((W'*A)./(W'*W*H + 1e-8));
        case 'h'
            H = H.*((W'*A)./(H*A'*W*H + 1e-8));
            for i = 1:1:k
                H(i,:) = H(i,:) ./ norm(H(i,:),2);
            end
            W = W.*((A*H')./(W*H*H'  + 1e-8));
        otherwise
            error('Invalid orthogonal constraint type.')
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