function F = kl_loss(A,W,H)
%%Papers:
% Kullback-Leibler Divergence for Nonnegative Matrix Factorization
% Z. Yang, H. Zhang, Z. Yuan, and E. Oja
%Inputs:
% A: Data matrix: n x m
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m
%Outputs
% F: Loss value

[n,k] = size(W);
[~,m] = size(H);

R = W*H + 1e-8;
F = sum(sum(A.*log(A ./ R))) + log(sum(sum(R)));

end