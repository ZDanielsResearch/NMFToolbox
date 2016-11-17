function [W,H] = nmft_mult(A,W,H,params)
%%Paper:
% Algorithms for Non-Negative Matrix Factorization
% D. Lee and H. Seung
%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
% params.maxIters: Maximum number of iterations to perform
% params.loss: Type of divergence to use: {'sqeuclidean','kldivergence','itakura-saito','alpha','beta'}
%Outputs
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m

[n,k] = size(W);
[~,m] = size(H);

if isempty(params.loss)
    params.loss = 'sqeuclidean';
    disp('Warning: No training loss specified: Using squared Euclidean.');
end

params.loss = lower(params.loss);

iterationNumber = 1;

switch params.loss
    case 'sqeuclidean'
        for iterationNumber = 1:1:params.maxIters;
            H = H.*((W'*A)./(W'*W*H + 1e-8));
            W = W.*((A*H')./(W*H*H'  + 1e-8));
        end
    case 'kldivergence'
        for iterationNumber = 1:1:params.maxIters;
            H = H.*(W'*(A./(W*H + 1e-8)))./(W'*ones(n,m));
            W = W.*((A./(W*H + 1e-8))*H')./(ones(n,m)*H');
        end
    otherwise
        error('Loss is not valid or incompatible with chosen method.');
end