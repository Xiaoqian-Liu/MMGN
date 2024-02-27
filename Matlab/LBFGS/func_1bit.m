function [obj, grad] = func_1bit(uv, Y, f, fprime, ind_omega)
% This function computes the objective value and gradient for the LBFGS method 
% in 1-bit matrix completion.

% --INPUTS-----------------------------------------------------------------------
% uv: the vectorized u and v
% Y: the observed binary matrix, unobserved entries are coded as zero
% f: the CDF
% fprime: the derivative of f
% ind_omega: the indicator vector of observations (column-major vectorization)

% --OUTPUTS-----------------------------------------------------------------------
% obj: the objective valus / loss
% grad: the gradient

% Xiaoqian Liu
% Dec. 2023

%%
    [d1, d2] = size(Y);
    r = length(uv)/(d1+d2);
    
    u = uv(1:(d1*r));
    v = uv((d1*r+1):end);
    
    U = reshape(u, [d1, r]);
    V = reshape(v, [d2, r]);
    M = U*V';
    
    %% compute the function value
    y = Y(ind_omega>0);
    d = (1+y)/2;
    m = M(ind_omega>0);% using conditioning is faster than indexing

    m1 = m(d>0);
    m0 = m(d==0);
    obj = -sum(log(f(m1))) - sum(log(1 - f(m0)));
    
    %% compute the gradient
    
    a = -y.*fprime(m)./f(y.*m);
    A = zeros(d1,d2);
    A(ind_omega>0) = a;
    A = sparse(A);
    %A = -Y.*fprime(M)./f(Y.*M); % here needs the unobserved entries to be 0

    G1 = A*V;
    G2 = A'*U;
    grad = [G1(:); G2(:)];
end