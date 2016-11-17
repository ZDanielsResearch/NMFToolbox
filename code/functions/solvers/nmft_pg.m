function [W,H] = nmft_pg(A,W,H,params)
%%Papers:
% Projected Gradient Methods for Nonnegative Matrix Factorization
% C.-J. Lin
% Optimality,  computation  and interpretation of nonnegative matrix factorizations
% M. Chu, F. Diele, R. Plemmons, and S. Ragni.
%Note: When run with 'newton' as step size, this is the same as ALS (but slower computationally).
%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
% params.maxIters: Maximum number of iterations to perform
% params.loss: Type of divergence to use: {'sqeuclidean','kldivergence','itakura-saito','alpha','beta'}
% params.stepType: How to compute step: {'steepest','newton'}
%Outputs
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m

if ~isempty(params.loss)
    params.loss = lower(params.loss);
    if ~strcmp(params.loss,'sqeuclidean')
        error('Loss must be squared Euclidean for projected gradient descent.');
    end
end

params.loss = lower(params.loss);

if isempty(params.stepType)
    params.stepType = 'steepest';
    disp('Warning: No step type specified: Using optimal steepest descent for sqeuclidean, Armijos rule otherwise.');
end

if strcmp(params.stepType,'newton')
    disp('Warning: Consider using ALS instead.');
end

params.stepType = lower(params.stepType);

% sH = [];
% sW = [];
% yH = [];
% yW = [];
% rhoH = [];
% rhoW = [];
% HH = [];
% HW = [];

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
        case 'lbfgs'
            [~,dH,~,alphaH,~,~,~] = sqeuclidean_loss(A,W,H,[1 0],[0 0]);
            H = H - alphaH .* dH;
            H = H .* (H > 0);
            [~,~,dW,~,alphaW,~,~] = sqeuclidean_loss(A,W,H,[0 1],[0 0]);
            W = W - alphaW .* dW;
            W = W .* (W > 0);
        otherwise
            error('Step type is not valid.')
    end
end

end
