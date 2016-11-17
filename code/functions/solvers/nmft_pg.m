function [W,H] = nmft_pg(A,W,H,params)
%%Papers:
% Projected Gradient Methods for Nonnegative Matrix Factorization
% C.-J. Lin
% Optimality,  computation  and interpretation of nonnegative matrix factorizations
% M. Chu, F. Diele, R. Plemmons, and S. Ragni.
%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
% params.maxIters: Maximum number of iterations to perform
% params.loss: Type of divergence to use: {'sqeuclidean','kldivergence','itakura-saito','alpha','beta'}
% params.stepType: How to compute step: {'steepest','newton','conjugate'}
%Outputs
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m

if isempty(params.stepType)
    params.stepType = 'steepest';
    disp('Warning: No step type specified: Using optimal steepest descent.');
end

params.stepType = lower(params.stepType);

iterationNumber = 1;

for iterationNumber = 1:1:params.maxIters
    switch params.stepType
        case 'steepest'
            [~,dH,~,alphaH,~,~,~] = sqeuclidean_loss(A,W,H,[1 0],[0 0]);
            H = H - alphaH .* dH;
            H = H .* (H > 0);
            [~,~,dW,~,alphaW,~,~] = sqeuclidean_loss(A,W,H,[0 1],[0 0]);
            W = W - alphaW .* dW;
            W = W .* (W > 0);
        case 'newton'
            [~,dH,~,~,~,alphaH,~] = sqeuclidean_loss(A,W,H,[1 0],[1 0]);
            H = H - inv(alphaH) * dH;
            H = H .* (H > 0);
            [~,~,dW,~,~,~,alphaW] = sqeuclidean_loss(A,W,H,[0 1],[0 1]);
            W = W - dW * inv(alphaW);
            W = W .* (W > 0);
        case 'conjugate'
            
        otherwise
            error('Step type is not valid.')
    end
    [F,~,~,~,~,~,~] = sqeuclidean_loss(A,W,H,[0 0],[0 0]);
    disp(F);
end
