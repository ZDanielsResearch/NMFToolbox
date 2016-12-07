function [A,W,H] = generate_data_matrix(rows,cols,rankVal,noiseLevel,positiveFlag,scale)
%%Inputs:
% rows: number of rows
% cols: number of columns
% rank: rank of underlying matrix: integer
% noiseLevel: coefficient for adding noise: scalar
% positiveFlag: does matrix only contain positive entries: boolean
% scale: scaling factor
%Outputs
% A: Synthetic data matrix

W = [];
for i = 1:1:rankVal
    x = scale .* rand(cols,1);
    if ~positiveFlag
        x = x - 0.5;
    end
    W = [W x];
end

H = rand(rankVal,rows);

A = W*H;
A = A';
 
A = A + noiseLevel .* rand(size(A));

A = A(randperm(size(A,1)),:);