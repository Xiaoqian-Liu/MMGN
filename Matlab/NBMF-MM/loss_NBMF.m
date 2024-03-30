function obj = loss_NBMF(D, U, V, A, B, omega)
% This function computes the objective value / loss of the NBMF

% --INPUTS-----------------------------------------------------------------------
% D: the Delta matrix in the objective function. Delta=(1+Y)/2, original Y
% U: the current estimate of the underlying matrix
% V: the index set of observations (column-major vectorization)
% A: matrix of alpha -1
% B: matrix of beta -1
% omega: the index of observations
% --OUTPUTS-----------------------------------------------------------------------
% obj: the objective valus / loss
    
% Xiaoqian Liu
% March. 2024

%%
    d = D(:);
    d = d(omega);
    
    M = U*V;
    m = M(:);
    m = m(omega);
    
    nOmega = length(omega);
    ix1 = find(d);  % find indices where d!=0 or equivalently d=1
    ix0 = setdiff(1:nOmega, ix1); % indices where d=0
    
    loss1 = -sum(log(m(ix1))) - sum(log(1 - m(ix0)));
    loss2 = -sum(A.*log(V) + B.*log(1-V), 'all');

    obj = (loss1+loss2)/nOmega;
end