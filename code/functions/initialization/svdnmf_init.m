function [W,H] = svdnmf_init(A,k)
%%Paper:
% New SVD based initialization strategy for non-negative matrix factorization
% H. Qiao
%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
%Outputs
% W: Basis matrix initialization: n x k
% H: Coefficient matrix initialization: k x m
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

[U, S, V] = svds(A, k);

W = abs(U(:,k));
H = abs(S(k,:)*V');

end