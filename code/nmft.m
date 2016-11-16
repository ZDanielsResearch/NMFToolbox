function [W,H,D] = nmft(A,k,method,maxIters,initialization)
%%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
% maxIters: Maximum number of iterations to perform
% initialization: How to initialize W and H: {'nndsvd','random','kmeans','svdnmf'}
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

if isempty(method)
    method = 'als';
end

if isempty(maxIters)
    maxIters = 50;
end

if isempty(initialization)
    initialization = 'nnsvd';
end

W = [];
H = [];
D = 0;

initialization = lower(initialization);
switch initialization
    case 'random'
        W = rand(n,k);
        H = rand(k,m);
    case 'kmeans'
        [W,H] = kmeansinit(A,k); 
    case 'nndsvd'
        [W,H] = nndsvd(A,k);
    case 'svdnmf'
        [W,H] = nmfsvd(A,k);
    otherwise
        error('Initialization is not valid.');
end

method = lower(method);
switch method
    case 'als'
        disp('blah!');
    otherwise
        error('Method is not valid.');
end

D = norm(W*H - A);

end

