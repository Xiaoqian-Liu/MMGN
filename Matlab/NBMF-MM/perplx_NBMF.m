function perplx = perplx_NBMF(D, U, V, omega)
% This function computes perplexity

% --INPUTS-----------------------------------------------------------------------
% D: the Delta matrix in the objective function. Delta=(1+Y)/2, original Y
% U: the current estimate of the underlying matrix
% V: the index set of observations (column-major vectorization)
% omega: the validation set
% --OUTPUTS-----------------------------------------------------------------------
% perplx: the perplexity
    

    d = D(:);
    d = d(omega);
    
    M = U*V;
    m = M(:);
    m = m(omega);
    
    nOmega = length(omega);
    ix1 = find(d);  % find indices where d!=0 or equivalently d=1
    ix0 = setdiff(1:nOmega, ix1); % indices where d=0
    
    perplx = -sum(log(m(ix1))) - sum(log(1 - m(ix0)));
    perplx = perplx/nOmega; 
end