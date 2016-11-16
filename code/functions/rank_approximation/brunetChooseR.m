function r = brunetChooseR(A,params,range,stepSize)
%%Paper:
% Metagenes and molecular pattern discovery using matrix factorization
% J. Brunet, P. Tamayo, T. Golub, and J. Mesirov
%Inputs:
% A: Data matrix: n x m
% method: Solver to use: {}
% range: [lower bound, upper bound] for rank
% stepSize: test nmf every step size units from lower bound to upper bound of rank
%Outputs
% r: Estimated rank of A