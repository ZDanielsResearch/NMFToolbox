function [W,H,FIters] = nmft_augmented(A,W,H,params)
%%%Augmented Lagrangian Method
%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
% params.maxIters: Maximum number of iterations to perform
% params.subIters: Number of subiterations to perform
% pgd2Flag: Instead of solving and then projecting, simultaneously solve and project
%Outputs
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m
% FIters: Sequence of function values

FIters = [];

if ~isempty(params.loss)
    params.loss = lower(params.loss);
    if ~strcmp(params.loss,'sqeuclidean')
        error('Loss must be squared Euclidean for projected gradient descent.');
    end
end
params.loss = lower(params.loss);

maxRho = 8000;

for iterationNumber = 1:1:params.maxIters
    numItersAugmented = 1;
    rho = 1;
    lambda = zeros(size(H));
    while true
        H1 = H;
        alpha = 1;
        numIters = 1;
        
        while true
            [F,dH,~,~,~,~,~] = sqeuclidean_loss(A,W,H1,[1 0],[0 0]);
            [Fnn, dHnn] = al_nn(H1,rho,lambda);
            F = F + Fnn;
            dH = dH + dHnn;
            Fold = F;

            while true
                H2 = H - alphaH .* dF;
                [F,~,~,~,~,~,~] = sqeuclidean_loss(A,W,H1,[0 0],[0 0]);
                [Fnn, ~] = al_nn(H1,rho,lambda);
                F = F + Fnn;
                if F <= Fold
                    H = H2;
                    break;
                else
                    alphaH = alphaH .* beta;
                end
            end
            
            numIters = numIters + 1;
            
            if abs(F - Fold) < 1e-3
                break;
            end
        end
        
        lambda = (lambda - 2.*rho.*H);
        lambda = lambda .* double(lambda > 0);
        if norm(H1 - H2,'fro') <= epsilon2 || mod(numItersAugmented,5) == 0
            rho = min(rho .* 2,maxRho);
        end
        
        [numItersAugmented norm(H1 - H2,'fro')]
        numItersAugmented = numItersAugmented + 1;
        
        if rho >= maxRho && (norm(H1 - H2,'fro') ./ norm(H1,'fro')) <= 1e-3
            break;
        end
    end
end