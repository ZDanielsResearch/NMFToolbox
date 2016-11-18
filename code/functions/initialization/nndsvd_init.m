function [W,H] = nndsvd_init(A,k)
%%Paper:
% SVD based initialization: A head start for nonnegative matrix factorization
% C. Boutsidis and E. Gallopoulos
%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
%Outputs
% W: Basis matrix initialization: n x k
% H: Coefficient matrix initialization: k x m
% D: Reconstruction error
%This code is a modified version of: https://raw.githubusercontent.com/jiyfeng/tfkld/master/matlab/nndSVD.m

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

k = k + 1;

[U, S, V] = svds(A, k);
W = zeros(n,k);
H = zeros(k,m);

W(:,1) = sqrt(S(1,1)).*U(:,1);
H(1,:) = sqrt(S(1,1)).*V(:,1)';

for i = 2:1:k
    x = U(:,i);
    y = V(:,i);
    xp = x .* (x >= 0); 
    xn = -x .* (x < 0);
    yp = y .* (y >= 0); 
    yn = -y .* (y < 0);
    xpnorm = norm(xp,2); 
    ypnorm = norm(yp,2); 
    mp = xpnorm .* ypnorm;
    xnnorm = norm(xn,2); 
    ynnorm = norm(yn,2); 
    mn = xnnorm .* ynnorm;
    if mp > mn
        u = xp ./ xpnorm;
        v = yp ./ ypnorm;
        sigma = mp;
    else
        u = xn/xnnorm;
        v = yn/ynnorm;
        sigma = mn;
    end
    W(:,i) = sqrt(S(i,i) .* sigma) .* u;
    H(i,:) = sqrt(S(i,i) .* sigma) .* v';
end

W = W(:,2:k);
H = H(2:k,:);

end