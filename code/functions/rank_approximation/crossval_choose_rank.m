function r = crossval_choose_rank(A,method,range,stepSize)
%%Inputs:
% A: Data matrix: n x m
% method: Solver to use: {}
% range: [lower bound, upper bound] for rank
% stepSize: test nmf every step size units from lower bound to upper bound of rank
%Outputs
% r: Estimated rank of A