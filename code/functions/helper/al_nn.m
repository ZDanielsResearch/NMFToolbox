function [Fnn, dFnn] = al_nn(A,rho,lambda)
%%%Helper function for NMF by Augmented Lagrangians

Fnn = ((lambda./(2.*rho)) - A);
Fnn1 = Fnn .* double(Fnn > 0);
Fnn = rho .* (Fnn1).^2;
Fnn = sum(sum(Fnn));

dFnn = -2.*rho.*Fnn1;

end

