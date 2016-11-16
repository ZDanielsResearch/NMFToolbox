function r = hutchinsChooseR(A,method,range,stepSize)
%%Paper: 
% Position-dependent motif characterization using non-negative matrix factorization
% L. Hutchins, S. Murphy, P. Singh, J. Graber
%Inputs:
% A: Data matrix: n x m
% method: Solver to use: {}
% range: [lower bound, upper bound] for rank
% stepSize: test nmf every step size units from lower bound to upper bound of rank
%Outputs
% r: Estimated rank of A