function [W,H] = normalize(W,H,normVal)
%Inputs/Outputs
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m
% normVal: Use L-normVal

S = zeros(size(W,2));
for i = 1:1:size(W,2)
    S(i,i) = norm(W(:,i),normVal);
    W(:,i) = W(:,i) ./ S(i,i);
end
Sinv = inv(S);
for i = 1:1:size(H,1)
    H(i,:) = H(i,:) ./ Sinv(i,i);
end

end