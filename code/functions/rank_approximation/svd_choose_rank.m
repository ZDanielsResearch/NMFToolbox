function r = svd_choose_rank(A,p)
%%Paper:
% New SVD based initialization strategy for non-negative matrix factorization
% H. Qiao
%Inputs:
% A: Data matrix: n x m
% p: Extracting proportion; Top-r singular values explain p percent of all singular values : Value between [0,1]
%Outputs
% r: Estimated rank of A

if isempty(A)
    error('Data matrix is empty.');
end

[n,m] = size(A);

if p > 1 || p < 0
    error('p must be between 0 and 1.');
end

[~,S,~] = svd(A);

s = diag(S);
sumTot = sum(s);

r = 0;

while true
    r = r + 1;
    sumR = sum(s(1:r));
    sumRplus1 = sum(s(1:r+1));
    if (sumR ./ sumTot) < p && (sumRplus1 ./ sumTot) >= p
        break;
    end
end

if (m + n) .* r > (m .* n)
    error('Rank r does not satisfy (m + n)r <= (mn) for selected p.');
end

end