function [W,H] = nmft_pg(A,W,H,params)
%%Papers:
% Projected Gradient Methods for Nonnegative Matrix Factorization
% C.-J. Lin
% Optimality,  computation  and interpretation of nonnegative matrix factorizations
% M. Chu, F. Diele, R. Plemmons, and S. Ragni.
%Note: When run with 'newton' as step size, this is the same as ALS (but slower computationally).
%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
% params.maxIters: Maximum number of iterations to perform
% params.stepType: How to compute step: {'steepest','newton','bfgs'}
% params.subIters: Number of subiterations to perform
%Outputs
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m

if ~isempty(params.loss)
    params.loss = lower(params.loss);
    if ~strcmp(params.loss,'sqeuclidean')
        error('Loss must be squared Euclidean for projected gradient descent.');
    end
end
params.loss = lower(params.loss);

if isempty(params.stepType)
    params.stepType = 'steepest';
    disp('Warning: No step type specified: Using optimal steepest descent.');
end

if strcmp(params.stepType,'newton')
    disp('Warning: Consider using ALS instead.');
end

params.stepType = lower(params.stepType);

if strcmp(params.stepType,'steepest') && (isempty(params.subIters) || params.subIters) < 1
    params.subIters = 1;
    disp('Warning: Invalid or missing number of subiterations specified: Using 1.');
end

if strcmp(params.stepType,'bfgs') && (isempty(params.subIters) || params.subIters < 1)
    params.subIters = 50;
    disp('Warning: Invalid or missing number of subiterations specified: Using 50.');
end


[n,k] = size(W);
[~,m] = size(H);

for iterationNumber = 1:1:params.maxIters
    switch params.stepType
        case 'steepest'
            for subIteration = 1:1:params.subIters
                [~,dH,~,alphaH,~,~,~] = sqeuclidean_loss(A,W,H,[1 0],[0 0]);
                H = H - alphaH .* dH;
            end
            H = H .* (H > 0);
            for subIteration = 1:1:params.subIters
                [~,~,dW,~,alphaW,~,~] = sqeuclidean_loss(A,W,H,[0 1],[0 0]);
                W = W - alphaW .* dW;
            end
            W = W .* (W > 0);
        case 'newton'
            [~,dH,~,~,~,alphaH,~] = sqeuclidean_loss(A,W,H,[1 0],[1 0]);
            H = H - inv(alphaH) * dH;
            H = H .* (H > 0);
            [~,~,dW,~,~,~,alphaW] = sqeuclidean_loss(A,W,H,[0 1],[0 1]);
            W = W - dW * inv(alphaW);
            W = W .* (W > 0);
        case 'bfgs'
            invHessH = eye(k,k);
            [F1,dH,~,~,~,~,~] = sqeuclidean_loss(A,W,H,[1 0],[0 0]);
            for subIteration = 1:1:params.subIters
                alphaH = 1;
                H1 = H;
                dH1 = dH;
                sameFlag = false;
                while true
                    H2 = H - alphaH .* invHessH * dH;
                    [F2,~,~,~,~,~,~] = sqeuclidean_loss(A,W,H2,[0 0],[0 0]);
                    if F2 < F1
                        H = H2;
                        break;
                    else
                        alphaH = alphaH .* 0.5;
                    end
                    if alphaH < 1e-6;
                        sameFlag = true;
                        break;
                    end
                end
                if sameFlag
                    break;
                end
                S = H - H1;
                [F1,dH,~,~,~,~,~] = sqeuclidean_loss(A,W,H,[1 0],[0 0]);
                Y = dH - dH1;
                rho = 1 ./ (Y(:)'*S(:));
                scaleMat = eye(k,k) - rho .* S*Y';
                invHessH = scaleMat*invHessH*scaleMat + rho*(S*S');
            end
            H = H .* (H > 0);
            
            invHessW = eye(k,k);
            [F1,~,dW,~,~,~,~] = sqeuclidean_loss(A,W,H,[0 1],[0 0]);
            for subIteration = 1:1:params.subIters
                alphaW = 1;
                W1 = W;
                dW1 = dW;
                sameFlag = false;
                while true
                    W2 = W - alphaW .* dW * invHessW;
                    [F2,~,~,~,~,~,~] = sqeuclidean_loss(A,W2,H,[0 0],[0 0]);
                    if F2 < F1
                        W = W2;
                        break;
                    else
                        alphaW = alphaW .* 0.5;
                    end
                    if alphaW < 1e-6;
                        sameFlag = true;
                        break;
                    end
                end
                if sameFlag
                    break;
                end
                S = W - W1;
                [F1,~,dW,~,~,~,d2W] = sqeuclidean_loss(A,W,H,[0 1],[0 0]);
                Y = dW - dW1;
                rho = 1 ./ (Y(:)'*S(:));
                scaleMat = eye(k,k) - rho .* S'*Y;
                invHessW = scaleMat*invHessW*scaleMat + rho*(S'*S);
            end
            W = W .* (W > 0);
        otherwise
            error('Step type is not valid.');
    end
    if params.printIter
        F = 0;
        if strcmp(params.evalLoss,'sqeuclidean')
            [F,~,~,~,~,~,~] = sqeuclidean_loss(A,W,H,[0 0],[0 0]);
        end
        if strcmp(params.evalLoss,'kldivergence')
            F = kl_loss(A,W,H);
        end
        disp(['Iteration #' num2str(iterationNumber) ', Function Value: ' num2str(F)]);
    end
end

end
