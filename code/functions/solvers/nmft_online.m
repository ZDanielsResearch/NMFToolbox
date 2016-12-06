function [W,H,FIters] = nmft_online(A,W,H,params,pgd2Flag)
%%Online Projected Gradient
%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
% params.maxIters: Maximum number of iterations to perform
% params.stepType: How to compute step: {'steepest','newton','bfgs'}
% params.subIters: Number of subiterations to perform
% params.sample: Percent of data points to sample
% pgd2Flag: Instead of solving and then projecting, simultaneously solve and project
%Outputs
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m
% FIters: Sequence of function values

FIters = [];

if params.printIter
    if strcmp(params.evalLoss,'sqeuclidean')
        [F,~,~,~,~,~,~] = sqeuclidean_loss(A,W,H,[0 0],[0 0]);
    end
    if strcmp(params.evalLoss,'kldivergence')
        F = kl_loss(A,W,H);
    end
    FIters = [FIters; F];
    disp(['Iteration #' num2str(0) ', Function Value: ' num2str(F)]);
end

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

if isempty(params.sample)
    params.sample = 0.1;
    disp('Warning: No sample size specified: Using 0.1.');
end

if params.sample <= 0 || params.sample >= 1
    disp('Invalid sample size.');
end

params.stepType = lower(params.stepType);

if strcmp(params.stepType,'steepest') && (isempty(params.subIters) || params.subIters) < 1
    params.subIters = 1;
    disp('Warning: Invalid or missing number of subiterations specified: Using 1.');
end

[n,k] = size(W);
[~,m] = size(H);

prevH = H;
prevW = W;
invHessH = eye(k,k);
invHessW = eye(k,k);

for iterationNumber = 1:1:params.maxIters
    sample = randperm(m);
    sample = sample(1:round(params.sample.*m));
    H2 = H(:,sample);
    A2 = A(:,sample);
    switch params.stepType
        case 'steepest'
            for subIteration = 1:1:params.subIters
                [~,dH,~,alphaH,~,~,~] = sqeuclidean_loss(A2,W,H2,[1 0],[0 0]);
                H2 = H2 - alphaH .* dH;
                if pgd2Flag
                    H2 = H2 .* (H2 > 0);
                end
            end
            if ~pgd2Flag
                H2 = H2 .* (H2 > 0);
            end
            for subIteration = 1:1:params.subIters
                [~,~,dW,~,alphaW,~,~] = sqeuclidean_loss(A2,W,H2,[0 1],[0 0]);
                W = W - alphaW .* dW;
                if pgd2Flag
                    W = W .* (W > 0);
                end
            end
            if ~pgd2Flag
                W = W .* (W > 0);
            end
        case 'mixed'
            [~,dH,~,~,~,alphaH,~] = sqeuclidean_loss(A2,W,H2,[1 0],[1 0]);
            H2 = H2 - inv(alphaH) * dH;
            H2 = H2 .* (H2 > 0);
            for subIteration = 1:1:params.subIters
                [~,~,dW,~,alphaW,~,~] = sqeuclidean_loss(A2,W,H2,[0 1],[0 0]);
                W = W - alphaW .* dW;
                if pgd2Flag
                    W = W .* (W > 0);
                end
            end
            if ~pgd2Flag
                W = W .* (W > 0);
            end
        case 'newton'
            [~,dH,~,~,~,alphaH,~] = sqeuclidean_loss(A2,W,H2,[1 0],[1 0]);
            H2 = H2 - inv(alphaH) * dH;
            H2 = H2 .* (H2 > 0);
            [~,~,dW,~,~,~,alphaW] = sqeuclidean_loss(A2,W,H2,[0 1],[0 1]);
            W = W - dW * inv(alphaW);
            W = W .* (W > 0);
        otherwise
            error('Step type is not valid.');
    end
    
    if params.printIter
        F = 0;
        H2 = zeros(size(H));
        startIndex = 1;
        for i = 0:params.sample:1
            endIndex = startIndex + ceil(params.sample.*m);
            endIndex = min(endIndex,m);
            A2 = A(:,startIndex:endIndex);
            H3 = inv(W'*W)*W'*A2;
            H3 = H3 .* (H3 >= 0);
            H2(:,startIndex:endIndex) = H3;
            startIndex = endIndex + 1;
        end
        if strcmp(params.evalLoss,'sqeuclidean')
            [F,~,~,~,~,~,~] = sqeuclidean_loss(A,W,H2,[0 0],[0 0]);
        end
        if strcmp(params.evalLoss,'kldivergence')
            F = kl_loss(A,W,H2);
        end
        FIters = [FIters; F];
        disp(['Iteration #' num2str(iterationNumber) ', Function Value: ' num2str(F)]);
    end
end

H2 = zeros(size(H));
startIndex = 1;
for i = 0:params.sample:1
    endIndex = startIndex + ceil(params.sample.*m);
    endIndex = min(endIndex,m);
    A2 = A(:,startIndex:endIndex);
    H3 = inv(W'*W)*W'*A2;
    H3 = H3 .* (H3 >= 0);
    H2(:,startIndex:endIndex) = H3;
    startIndex = endIndex + 1;
end
H = H2;

end
