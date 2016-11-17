function [F,dH,dW,alphaH,alphaW,d2H,d2W] = sqeuclidean_loss(A,W,H,computeDF,computeD2F)
%Inputs:
% A: Data matrix: n x m
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m
% computeDF: Flags for whether the gradient of the loss should be computed:
% computeDF = [dHFlag, dWFlag] where dHFlag and dWFlag are 0 (do not
% compute respective gradient) or 1 (compute respective gradient).
% computeD2F = [d2HFlag, d2WFlag] where d2HFlag and d2WFlag are 0 (do not
% compute respective Hessian) or 1 (compute respective Hessian).
%Outputs
% F: Loss value
% dF: Gradient of loss

F = (A-W*H);
F = trace(F*F');
dH = [];
dW = [];
d2H = [];
d2W = [];
alphaW = 0;
alphaH = 0;

if computeDF(1) || computeD2F(1)
    dH = 2.*W'*(W*H - A);
    alphaH = trace(H'*W'*W*dH - dH'* W'*A) ./ trace(dH'*W'*W*dH);
end

if computeDF(2) || computeD2F(2)
    dW = 2.*(W*H - A)*H';
    alphaW = trace(H'*dW'*W*H - A'*dW*H) ./ trace(H'*dW'*dW*H);
end

if computeD2F(1)
    d2H = 2.*W'*W;
end

if computeD2F(2)
    d2W = 2.*H*H';
end

end