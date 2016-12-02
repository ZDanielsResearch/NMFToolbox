function [W,H,FIters] = nmft_al(A,W,H,params)
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

maxRho = 1024;
beta = 0.8;

for iterationNumber = 1:1:params.maxIters
    numItersAugmented = 1;
    rho = 1;
    lambda = zeros(size(H));
    
    while true
        H1 = H;
        alphaH = 1;
        numIters = 1;
        
        for subIteration = 1:1:params.subIters
            [F,dH,~,~,~,~,~] = sqeuclidean_loss(A,W,H,[1 0],[0 0]);
            [Fnn, dHnn] = al_nn(H,rho,lambda);
            F = F + Fnn;
            dH = dH + dHnn;
            Fold = F;

            while true
                H2 = H - alphaH .* dH;
                [F,~,~,~,~,~,~] = sqeuclidean_loss(A,W,H2,[0 0],[0 0]);
                [Fnn, ~] = al_nn(H2,rho,lambda);
                F = F + Fnn;
                if F <= Fold
                    H = H2;
                    break;
                else
                    alphaH = alphaH .* beta;
                end
            end
            
            numIters = numIters + 1;
                                    
            if Fold - F <= 1e-3
                break;
            end
        end
                
        lambda = (lambda - 2.*rho.*H);
        lambda = lambda .* double(lambda > 0);
                
        if (norm(H1 - H2,'fro') ./ norm(H1,'fro')) <= 1e-2 || mod(numItersAugmented,5) == 0
            rho = min(rho .* 2,maxRho);
        end
        
        numItersAugmented = numItersAugmented + 1;
        
        if (rho >= maxRho && (norm(H1 - H2,'fro') ./ norm(H1,'fro')) <= 1e-2) || (numItersAugmented > params.subIters) 
            break;
        end
    end
    
    H = H .* double(H > 0);
    
    numItersAugmented = 1;
    rho = 1;
    lambda = zeros(size(W));
    
    while true
        W1 = W;
        alphaW = 1;
        numIters = 1;
        
        for subIteration = 1:1:params.subIters
            [F,~,dW,~,~,~,~] = sqeuclidean_loss(A,W,H,[0 1],[0 0]);
            [Fnn, dWnn] = al_nn(W,rho,lambda);
            F = F + Fnn;
            dW = dW + dWnn;
            Fold = F;

            while true
                W2 = W - alphaW .* dW;
                [F,~,~,~,~,~,~] = sqeuclidean_loss(A,W2,H,[0 0],[0 0]);
                [Fnn, ~] = al_nn(W2,rho,lambda);
                F = F + Fnn;
                if F <= Fold
                    W = W2;
                    break;
                else
                    alphaW = alphaW .* beta;
                end
            end
            
            numIters = numIters + 1;
                                    
            if Fold - F <= 1e-3
                break;
            end
        end
                
        lambda = (lambda - 2.*rho.*W);
        lambda = lambda .* double(lambda > 0);
                
        if (norm(W1 - W2,'fro') ./ norm(W1,'fro')) <= 1e-2 || mod(numItersAugmented,5) == 0
            rho = min(rho .* 2,maxRho);
        end
        
        numItersAugmented = numItersAugmented + 1;
        
        if (rho >= maxRho && (norm(W1 - W2,'fro') ./ norm(W1,'fro')) <= 1e-2) || (numItersAugmented > params.subIters) 
            break;
        end
    end
    
    W = W .* double(W > 0);
    
    if params.printIter
        F = 0;
        if strcmp(params.evalLoss,'sqeuclidean')
            [F,~,~,~,~,~,~] = sqeuclidean_loss(A,W,H,[0 0],[0 0]);
        end
        if strcmp(params.evalLoss,'kldivergence')
            F = kl_loss(A,W,H);
        end
        FIters = [FIters; F];
        disp(['Iteration #' num2str(iterationNumber) ', Function Value: ' num2str(F)]);
    end
end