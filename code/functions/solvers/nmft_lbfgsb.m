function [W,H] = nmft_lbfgsb(A,W,H,params)
%%Papers:
% A limited memory algorithm for bound constrained optimization
% R. Byrd, P. Lu, J. Nocedal, C. Zhu
%Inputs:
% A: Data matrix: n x m
% k: Number of basis elements
% params.maxIters: Maximum number of iterations to perform
% params.loss: Type of divergence to use: {'sqeuclidean','kldivergence','itakura-saito','alpha','beta'}
%Outputs
% W: Basis matrix: n x k
% H: Coefficient matrix: k x m

if ~isempty(params.loss)
    params.loss = lower(params.loss);
    if ~strcmp(params.loss,'sqeuclidean')
        error('Loss must be squared Euclidean for L-BFGS-B.');
    end
end



for iterationNumber = 1:1:params.maxIters    
    w = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
    h = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
end

end
