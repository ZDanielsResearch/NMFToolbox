function [W,H,FIters] = nmft_linear(data,W,H,params)
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

[n,k] = size(W);
[~,m] = size(H);

X = data;
for i = 1:1:m
    X(:,i) = X(:,i) ./ sum(X(:,i));
end

% index = zeros(m,m);
% count = 1;
% for i = 1:1:m
%     for j = 1:1:m
%         index(j,i) = count;
%         count = count + 1;
%     end
% end
% numElements = count - 1;
%
% C = zeros(n,n);
% p = rand(1,n);
% beta = 0;
% sp = 0.1;
% sd = 0.01;
% 
% 
% diagElements = diag(index);
% p1 = rand([1,length(diagElements)]);
% p = zeros(1,numElements);
% for i = 1:1:length(diagElements)
%     p(diagElements(i)) = p1(i);
% end
% 
% epsilon = 10;
% 
% A = zeros(m^2 + 2.*m + 1,numElements);
% b = zeros(m^2 + 2.*m + 1,1);
% lb = zeros(numElements,1);
% count = 1;
% for i = 1:1:m
%     for j = 1:1:m
%         if i == j
%             A(count,index(i,j)) = 1;
%             b(count) = 1;
%         else
%             A(count,index(i,j)) = 1;
%             A(count,index(i,i)) = -1;
%             b(count) = 0;
%         end
%         count = count + 1;
%     end
%     normM = sum(X(:,i));
%     coefficients = sum(X);
%     
%     
%     A(count,index(:,i)) = -1;
%     b(count) = 2*epsilon - 1;
%     count = count + 1;
%     
%     A(count,index(:,i)) = 1;
%     b(count) = 2*epsilon + 1;
%     count = count + 1;
% end
% A(count,diagElements) = 1;
% b(count) = k;
% 
% c = linprog(p,A,b,[],[],lb,[]);
% C = reshape(c,[m,m]);
% 
% [~,selected] = sort(diag(C2),'descend');
% H = data(selected(1:k),:);
% W = data*H'*inv(H*H');
% W = W .* (W >= 0);

p = rand(1,m);
epsilon = 1;

warning('off');
cvx_begin
    variable C(m,m) 
    minimize( p*diag(C) )
    subject to
        C(:) >= 0; 
        for i = 1 : m
            norm(X(:,i) - X*C(:,i),1) <= 2*epsilon; 
            C(i,i) <= 1; 
            for j = 1 : m
                C(i,j) <= C(i,i);
            end
        end
        trace(C) == k; 
cvx_end
warning('on');

[~,selected] = sort(diag(C),'descend');
H = data(selected(1:k),:);
W = data*H'*inv(H*H');
W = W .* (W >= 0);

end
