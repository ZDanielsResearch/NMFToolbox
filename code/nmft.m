function [W,H,D,F,FIters] = nmft(A,k,params)
%%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
% params.method: Solver to use: {'als','acls','ahcls','gdcls','mult','projgrad','projgrad2','nnsc','hoyer','orthoChoi','orthoDTPP','convex','augmented','linear','online','online2'}
% params.maxIters: Maximum number of iterations to perform
% params.initialization: How to initialize W and H: {'nndsvd','random','kmeans','svdnmf'}
% params.loss: Type of divergence to use for training: {'sqeuclidean','kldivergence','itakura-saito','alpha','beta'}
% params.evalLoss: Type of divergence to use for evaluation: {'sqeuclidean','kldivergence','itakura-saito','alpha','beta'}
% params.stepType: How to compute step: {'steepest','newton','bfgs','mixed'}
% params.paramH: parameter associated with H: Differs from algorithm to algorithm
% params.paramW: parameter associated with W: Differs from algorithm to algorithm
% params.sparseParamH: parameter for Hoyer sparsity associated with H
% params.sparseParamW: parameter for Hoyer sparsity associated with W
% params.orthogonalConstraint: Enforce orthogonality in {'w','h'} for Choi algorithm
% params.subIters: Number of subiterations to perform
% params.printIter: Flag to print function value after each iteration: {true,false}
% params.sample: Percent of data points to sample
%Outputs
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m
% D: Reconstruction error
% F: Positive convex combination coefficient matrix for convex NMF
% FIters: Sequence of function values

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

if isempty(params.printIter)
    params.printIter = false;
end

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
params.method = lower(params.method);

W = [];
H = [];
F = [];
FIters = [];

if sum(sum(A < 0)) == 0
    params.initialization = lower(params.initialization);
    switch params.initialization
        case 'random'
            W = abs(rand(n,k));
            H = abs(rand(k,m));
        case 'kmeans'
            [W,H] = kmeans_init(A,k); 
        case 'nndsvd'
            [W,H] = nndsvd_init(A,k);
        case 'svdnmf'
            [W,H] = svdnmf_init(A,k);
        otherwise
            error('Initialization is not valid.');
    end
else
    if ~strcmp(params.method,'convex') && ~strcmp(params.method,'semi')
        error('Data matrix has negative entries and method is not Convex-NMF or Semi-NMF');
    end
    W = zeros(n,k);
    H = zeros(k,m);
end

switch params.method
    case 'als'
        [W,H,FIters] = nmft_als(A,W,H,params);
    case 'acls'
        [W,H,FIters] = nmft_acls(A,W,H,params);
    case 'ahcls'
        [W,H,FIters] = nmft_ahcls(A,W,H,params);
    case 'gdcls'
        [W,H,FIters] = nmft_gdcls(A,W,H,params);
    case 'mult'
        [W,H,FIters] = nmft_mult(A,W,H,params);
    case 'projgrad'
        [W,H,FIters] = nmft_pg(A,W,H,params,0);
    case 'projgrad2'
        [W,H,FIters] = nmft_pg(A,W,H,params,1);
    case 'nnsc'
        [W,H,FIters] = nmft_nnsc(A,W,H,params);
    case 'hoyer'
        [W,H,FIters] = nmft_hoyer(A,W,H,params);
    case 'orthochoi'
        [W,H,FIters] = nmft_orthogonal_choi(A,W,H,params);
    case 'orthodtpp'
        [W,H,FIters] = nmft_orthogonal_dtpp(A,W,H,params);
    case 'convex'
        [W,H,F,FIters] = nmft_convex(A,W,H,params);
    case 'semi'
        [W,H,FIters] = nmft_semi(A,W,H,params);
    case 'augmented'
        [W,H,FIters] = nmft_augmented(A,W,H,params);
    case 'linear'
        [W,H,FIters] = nmft_linear_2(A,W,H,params);
    case 'online'
        [W,H,FIters] = nmft_online(A,W,H,params,0);
    case 'online2'
        [W,H,FIters] = nmft_online(A,W,H,params,1);
    otherwise
        error('Method is not valid.');
end

D = 0;

switch params.evalLoss
    case 'sqeuclidean'
        [D,~,~,~,~,~,~] = sqeuclidean_loss(A,W,H,[0 0],[0 0]);
    case 'kldivergence'
        D = kl_loss(A,W,H);
    otherwise
        error('Loss is not valid.');
end

end

