function obj = loss_1bit(D, M, omega, f)
% This function computes the objective value / loss of the MMGN method for 
% 1-bit matrix completion.

% --INPUTS-----------------------------------------------------------------------
% D: the Delta matrix in the objective function. Delta=(1+Y)/2
% M: the current estimate of the underlying matrix
% omega: the index set of observations (column-major vectorization)
% f: the CDF of the noise (probit or logistic)

% --OUTPUTS-----------------------------------------------------------------------
% obj: the objective valus / loss

    d = D(:);
    d = d(omega);
    
    m = M(:);
    m = m(omega);
    
    nOmega = length(omega);
    ix1 = find(d);  % find indices where d!=0 or equivalently d=1
    ix0 = setdiff(1:nOmega, ix1); % indices where d=0
    
    obj = -sum(log(f(m(ix1)))) - sum(log(1 - f(m(ix0))));
end