function [F,dW,dH] = sqeuclidean_loss(A,W,H,computeDF)
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

F = (A-W*H);
F = trace(F*F');
dH = [];
dW = [];

if computeDF(1) == 1
    dH = 2.*W'*(W*H - A);
end

if computeDF(2) == 1
    dW = 2.*(W*H - A)*H';
end

end