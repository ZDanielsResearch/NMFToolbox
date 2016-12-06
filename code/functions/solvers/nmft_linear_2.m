function [W,H,FIters] = nmft_linear(A,W,H,params)
%%Note this implementation might be incorrect.
%%Papers:
% Factoring nonnegative matrices with linear programs
% V. Bittorf, B. Recht, C. Re, J. Tropp
%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
% params.maxIters: Maximum number of iterations to perform
%Outputs
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m
% FIters: Sequence of function values

FIters = [];

if ~isempty(params.loss)
    params.loss = lower(params.loss);
    if ~strcmp(params.loss,'sqeuclidean')
        error('Loss must be squared Euclidean for Hottopixx.');
    end
end
params.loss = lower(params.loss);

[n,r] = size(W);
[~,m] = size(H);

X = A;
for i = 1:1:m
    X(:,i) = X(:,i) ./ sum(X(:,i));
end

C = zeros(n,n);
p = rand(n,1);
beta = 0;
sp = 0.1;
sd = 0.01;

for iterationNumber = 1:1:params.maxIters
    iterationNumber
    for i = 1:1:m
        k = randi(m);
        mu = sum(X ~= 0,2) ./ m;
        C = C + sp.*sign(X(:,k)-C*X(:,k))*X(:,k)' - sp.*diag(mu.*(beta.*ones(size(p)).*p));
    end
    for i = 1:1:n
        z = C(:,i);
        [z,ind] = sort(z,'descend');
        unsorted = 1:length(z);
        newInd(ind) = unsorted;
        mu2 = z(1);
        jc = 1;
        for j = 2:1:r
            pmu = mu2;
            if mu2 > 1
                mu2 = 1;
            elseif mu2 < 0
                mu2 = 0;
            end
            if z(j) <= pmu
                jc = j - 1;
            else
                mu2 = ((j-1)./j).*mu2 + (1./j).*z(j);
            end
        end
        pmu = mu2;
        if mu2 > 1
            mu2 = 1;
        elseif mu2 < 0
            mu2 = 0;
        end
        x = zeros(size(z));
        x(1) = pmu;
        for j = 1:1:jc
            x(j) = pmu;
        end
        for j = jc+1:1:r
            x(j) = max(x(j),0);
        end
        x = x(newInd);
        C(:,i) = x;
    end
    
    beta = beta + sd .*(trace(C) - r);
end
[~,selected] = sort(diag(C),'descend');
H = A(selected(1:r),:);
W = A*H'*inv(H*H');
W = W .* (W >= 0);

end
