function [W,H] = nmft_als(A,W,H,params)
%%Paper:
% Positive matrix factorization: A non-negative factor model with optimal utilization of error estimates of data values
% P. Paatero and U. Tapper
%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
% params.maxIters: Maximum number of iterations to perform
% params.loss: Type of divergence to use: {'sqeuclidean','kldivergence','itakura-saito','alpha','beta'}
%Outputs
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m

if ~isempty(params.loss)
    params.loss = lower(params.loss);
    if ~strcmp(params.loss,'sqeuclidean')
        error('Loss must be squared Euclidean for alternating least squares.');
    end
end

for iterationNumber = 1:1:params.maxIters
    H = inv(W'*W)*W'*A;
    H = H .* (H >= 0);
    W = A*H'*inv(H*H');
    W = W .* (W >= 0);
    [F,~,~,~,~,~,~] = sqeuclidean_loss(A,W,H,[0 0],[0 0])
end

end