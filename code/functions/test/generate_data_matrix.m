function A = generate_data_matrix(rows,cols,rankVal,noiseLevel,positiveFlag)
%%Inputs:
% rows: number of rows
% cols: number of columns
% rank: rank of underlying matrix: integer
% noiseLevel: coefficient for adding noise: scalar
% positiveFlag: does matrix only contain positive entries: boolean 
%Outputs
% A: Synthetic data matrix

A = [];
for i = 1:1:rankVal
    x = rand(1,cols);
    if ~positiveFlag
        x = x - 0.5;
    end
    A = [A; x];
end

for i = rankVal+1:1:rows
    i
    x = zeros(1,cols);
    continueFlag = true;
    while continueFlag
        rowIndex = randi(rankVal);
        coefficient = rand(1,1);
        x = x + coefficient .* A(rowIndex,:);
        continueFlag = rand(1,1) <= 0.5;
    end
    A = [A; x];
end

A = A + noiseLevel .* rand(size(A));

A = A(randperm(size(A,1)),:);