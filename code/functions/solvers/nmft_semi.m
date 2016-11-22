function [W,H,FIters] = nmft_semi(A,W,H,params)
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
% FIters: Sequence of function values

FIters = [];

[n,k] = size(W);
[~,m] = size(H);

if ~isempty(params.loss)
    params.loss = lower(params.loss);
    if ~strcmp(params.loss,'sqeuclidean')
        error('Loss must be squared Euclidean for semi-nonnegative matrix factorization.');
    end
end

H = rand(m,k);
W = A*H*inv(H'*H);

for iterationNumber = 1:1:params.maxIters;
    B = (A'*W);
    Bp = (abs(B)+B)./2;
    Bn = (abs(B)-B)./2;
    
    C = (W'*W);
    Cp = (abs(C)+C)./2;
    Cn = (abs(C)-C)./2;
    
    H = H .* ((Bp + H*Cn) ./ (Bn + H*Cp + 1e-8)).^(0.5);
    
    W = A*H*inv(H'*H);
    
    if params.printIter
        if strcmp(params.evalLoss,'sqeuclidean')
            [F1,~,~,~,~,~,~] = sqeuclidean_loss(A,W,H',[0 0],[0 0]);
        end
        if strcmp(params.evalLoss,'kldivergence')
            F1 = kl_loss(A,W,H');
        end
        FIters = [FIters; F1];
        disp(['Iteration #' num2str(iterationNumber) ', Function Value: ' num2str(F1)]);
    end
end
H = H';