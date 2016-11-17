function [W,H] = nmft_ahcls(A,W,H,params)
%%Paper:
% Algorithms, Initializations, and Convergence for the Nonnegative Matrix Factorization
% A. Langville, C. Meyer, R. Albright, J. Cox, and D. Duling
%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
% params.maxIters: Maximum number of iterations to perform
% params.loss: Type of divergence to use: {'sqeuclidean','kldivergence','itakura-saito','alpha','beta'}
% params.paramH: mixing coefficient associated with H
% params.paramW: mixing coefficient associated with W
% params.sparseParamH: parameter for Hoyer sparsity associated with H
% params.sparseParamW: parameter for Hoyer sparsity associated with W
%Outputs
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m

if ~isempty(params.loss)
    params.loss = lower(params.loss);
    if ~strcmp(params.loss,'sqeuclidean')
        error('Loss must be squared Euclidean for alternating Hoyer constrained least squares.');
    end
end

if isempty(params.paramH)
    params.paramH = 0;
    disp('Warning: No params.paramH specified: Using params.paramH = 0.');
end

if isempty(params.paramW)
    params.paramW = 0;
    disp('Warning: No params.paramW specified: Using params.paramW = 0.');
end

if isempty(params.sparseParamH)
    params.sparseParamH = 0;
    disp('Warning: No params.sparseParamH specified: Using params.sparseParamH = 0.');
end

if isempty(params.sparseParamW)
    params.sparseParamW = 0;
    disp('Warning: No params.sparseParamW specified: Using params.sparseParamW = 0.');
end

[n,k] = size(W);
[~,m] = size(H);

betaH = ((1 - params.sparseParamH).*sqrt(k) + params.sparseParamH).^2;
betaW = ((1 - params.sparseParamW).*sqrt(k) + params.sparseParamW).^2;

for iterationNumber=1:1:params.maxIters
    H = inv(W'*W + params.paramH .* betaH .* eye(size(k,k)) - params.paramH .* ones(size(k,k)))*W'*A;
    H = H .* (H >= 0);
    W = A*H'*inv(H*H' + params.paramW .* betaW .* eye(k,k) - params.paramW .* ones(size(k,k)));
    W = W .* (W >= 0);
end

end