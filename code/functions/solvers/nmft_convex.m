function [W,H,F] = nmft_convex(A,W,H,params)
%%Paper:
% Convex and Semi-Negative Matrix Factorizations
% C. Ding, T. Li, M. Jordan
%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
% params.maxIters: Maximum number of iterations to perform
%Outputs
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m

[n,k] = size(W);
[~,m] = size(H);

if ~isempty(params.loss)
    params.loss = lower(params.loss);
    if ~strcmp(params.loss,'sqeuclidean')
        error('Loss must be squared Euclidean for convex nonnegative matrix factorization.');
    end
end

H = H';

% C = kmeans(A',k);
% for i = 1:1:k
%     H(:,i) = double(C == k);
% end

H = rand(m,k);
H = H + 0.2.*ones(m,k);
F = H*inv(diag(sum(H)));
F = F .* (F > 0);

for iterationNumber = 1:1:params.maxIters;
    B = (A'*A);
    Bp = (abs(B)+B)./2;
    Bn = (abs(B)-B)./2;
    H = H .* (((Bp*F) + (H*F'*Bn*F)) ./ ((Bn*F) + (H*F'*Bp*F) + 1e-8)).^(0.5);
    F = F .* (((Bp*H) + (Bn*F*H'*H)) ./ ((Bn*H) + (Bp*F*H'*H) + 1e-8)).^(0.5);
    if params.printIter
        W = A*F;
        if strcmp(params.evalLoss,'sqeuclidean')
            [F1,~,~,~,~,~,~] = sqeuclidean_loss(A,W,H',[0 0],[0 0]);
        end
        if strcmp(params.evalLoss,'kldivergence')
            F1 = kl_loss(A,W,H');
        end
        disp(['Iteration #' num2str(iterationNumber) ', Function Value: ' num2str(F1)]);
    end
end
H = H';
W = A*F;