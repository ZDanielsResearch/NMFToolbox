function [W,H,D] = nmft(A,k,params)
%%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
% params.method: Solver to use: {'als','acls','ahcls','mult','projgrad'}
% params.maxIters: Maximum number of iterations to perform
% params.initialization: How to initialize W and H: {'nndsvd','random','kmeans','svdnmf'}
% params.loss: Type of divergence to use for training: {'sqeuclidean','kldivergence','itakura-saito','alpha','beta'}
% params.evalLoss: Type of divergence to use for evaluation: {'sqeuclidean','kldivergence','itakura-saito','alpha','beta'}
% params.stepType: How to compute step: {'steepest','newton','lbfgs'}
% params.paramH: parameter associated with H: Differs from algorithm to algorithm
% params.paramW: parameter associated with W: Differs from algorithm to algorithm
% params.sparseParamH: parameter for Hoyer sparsity associated with H
% params.sparseParamW: parameter for Hoyer sparsity associated with W
%Outputs
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m
% D: Reconstruction error

if isempty(A)
    error('Data matrix is empty.');
end

[n,m] = size(A);

if isempty(k) || k <= 0
    error('Number of basis elements is empty or zero.');
end

if k > n || k > m
    error('Number of basis elements is larger than the number of rows or columns in data matrix.');
end

if isempty(params.method)
    params.method = 'als';
    disp('Warning: No method specified: Using alternating least squares for solving.');
end

if isempty(params.maxIters)
    params.maxIters = 50;
    disp('Warning: No maximum number of iterations specified: Using 50 iterations.');
end

if isempty(params.initialization)
    params.initialization = 'nndsvd';
    disp('Warning: No initialization method specified: Using NNDSVD.');
end

W = [];
H = [];

params.initialization = lower(params.initialization);
switch params.initialization
    case 'random'
        W = abs(rand(n,k));
        H = abs(rand(k,m));
    case 'kmeans'
        [W,H] = kmeansinit(A,k); 
    case 'nndsvd'
        [W,H] = nndsvd(A,k);
    case 'svdnmf'
        [W,H] = svdnmf(A,k);
    otherwise
        error('Initialization is not valid.');
end

params.method = lower(params.method);
switch params.method
    case 'als'
        [W,H] = nmft_als(A,W,H,params);
    case 'acls'
        [W,H] = nmft_acls(A,W,H,params);
    case 'ahcls'
        [W,H] = nmft_ahcls(A,W,H,params);
    case 'mult'
        [W,H] = nmft_mult(A,W,H,params);
    case 'projgrad'
        [W,H] = nmft_pg(A,W,H,params);
    otherwise
        error('Method is not valid.');
end

D = 0;

if isempty(params.evalLoss)
    if isempty(params.loss)
        params.evalLoss = 'sqeuclidean';
        disp('Warning: No evaluation or training loss specified: Using squared euclidean loss for evaluation.');
    else
        params.evalLoss = params.loss;
        disp('Warning: No evaluation loss specified: Using training loss for evaluation.');
    end
end

params.evalLoss = lower(params.evalLoss);
switch params.evalLoss
    case 'sqeuclidean'
        [D,~,~,~,~,~,~] = sqeuclidean_loss(A,W,H,[0 0],[0 0]);
    case 'kldivergence'
        D = kl_loss(A,W,H);
    otherwise
        error('Loss is not valid.');
end

end

