function [W,H] = kmeans_init(A,k)
%%Paper:
% Concept Decompositions for Large Sparse Text Data using Clustering
% I. Dhillon and D. Modha
%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
%Outputs
% W: Basis matrix initialization: n x k
% H: Coefficient matrix initialization: k x m
% D: Reconstruction error

[~,W] = kmeans(A',k);
W = W';

for i = 1:1:k
	W(:,i) = W(:,i) ./ norm(W(:,i),2);
end

H = inv(W'*W)*W'*A;
H = H .* (H >= 0);

end