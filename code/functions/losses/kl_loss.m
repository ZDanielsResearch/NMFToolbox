function [F,dW,dH] = kl_loss(A,W,H,computeDF)
%Inputs:
% A: Data matrix: n x m
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m
% computeDF: Flags for whether the gradient of the loss should be computed:
% computeDF = [dHFlag, dWFlag] where dHFlag and dWFlag are 0 (do not
% compute respective gradient) or 1 (compute respective gradient).
%Outputs
% F: Loss value
% dF: Gradient of loss

[n,k] = size(W);
[~,m] = size(H);

R = W*H;
F = sum(sum(A.*log(A ./ (R + 1e-8))) + log(sum(sum(R)) + 1e-8));
dH = [];
dW = [];

if computeDF(1) || computeDF(2)
    Z = (A./R);
end

if computeDF(1)
    dH = -W'*Z + W'*ones(n,m);
end

if computeDF(2)
    dW = -Z*H' + ones(n,m)*H';
end

end