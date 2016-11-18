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

for iterationNumber = 1:1:params.maxIters;
    switch params.loss
        case 'sqeuclidean'
            H = H.*((W'*A)./(W'*W*H + 1e-8));
            W = W.*((A*H')./(W*H*H'  + 1e-8));
        case 'kldivergence'
            H = H.*(W'*(A./(W*H + 1e-8)))./(W'*ones(n,m));
            W = W.*((A./(W*H + 1e-8))*H')./(ones(n,m)*H');
        otherwise
            error('Loss is not valid or incompatible with chosen method.');
    end
    if params.printIter
        F = 0;
        if strcmp(params.evalLoss,'sqeuclidean')
            [F,~,~,~,~,~,~] = sqeuclidean_loss(A,W,H,[0 0],[0 0]);
        end
        if strcmp(params.evalLoss,'kldivergence')
            F = kl_loss(A,W,H);
        end
        disp(['Iteration #' num2str(iterationNumber) ', Function Value: ' num2str(F)]);
    end
end