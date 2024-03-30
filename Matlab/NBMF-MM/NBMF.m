function [U, V, obj, relchange] = NBMF(D, omega, R, varargin)
% This function translate the NBMF algorithm in  
% ''A majorization-minimization algorithm for nonnegative binary matrix
% factorization''. Original python code in https://github.com/magronp/NMF-binary.


% --INPUTS-----------------------------------------------------------------------
% D: observed binary data matrix, zero-one
% omega: the index set of observations (column-major vectorization)
% R: target rank in the rank constraint
% varargin: additional parameters, including the following
%       tol: the tolerance for the relative change in the estimate or
%            objective value (for early stopping)
%       maxiters: the maximum number of iterations
%       alpha: prior parameter for H, default is 1
%       beta: prior parameter for H, default is 1


% --OUTPUTS-----------------------------------------------------------------------
% U: Mhat = U*V, U is nonnegative
% V: Mhat = U*V, V is nonnegative
% relchange: the sequence of the ralative change in the estimate/objective
%            for early stopping
% obj: the sequence of objective values at each iteration

% Xiaoqian Liu
% March. 2024

%% Set algorithm parameters from input or by using defaults.
params = inputParser;
params.addParameter('alpha', 1, @isscalar);
params.addParameter('beta', 1, @isscalar);
params.addParameter('tol', 1e-6, @isscalar);
params.addParameter('maxiters', 2e3, @(x) isscalar(x) & x > 0);
params.parse(varargin{:});

%% Copy from params object.
alpha = params.Results.alpha;
beta = params.Results.beta;
tol = params.Results.tol;
maxiters = params.Results.maxiters;


D(isnan(D)) = 0;
[m, n] = size(D); % D is zero or one
%% Initialization
rng(2022);
U = rand(m, R);
U = U./sum(U);
V = rand(R, n);

% additional matrices
A = ones(R, n)*(alpha-1);
B = ones(R, n)*(beta-1);

loss_last = -inf;


obj = nan(maxiters, 1);
relchange = nan(maxiters, 1);

for iter = 1:maxiters
    
    UV = U*V;
    a = D./(UV+eps);
    b = (1-D)./(1-UV+eps);
    numerator = V.*(U'*a)+A;
    denom = (1-V).*(U'*b) +B;
    % update on H/V
    V = numerator ./ (numerator+denom);
    
    % updates
    UV1 = U*V;
    a1 = D./(UV1+eps);
    b1 = (1-D)./(1-UV1+eps);
    % update on W/U
    U = U.*( a1*V' + b1*(1-V)')/n;
    
    loss_new = loss_NBMF(D, U, V, A, B, omega);
    obj(iter) = loss_new;
    relchange(iter) = abs( (loss_last-loss_new) / (loss_last+eps) );
    if (relchange(iter)< tol )
        break;
    end
    
    loss_last = loss_new;
   
end

% outputs
obj = obj(1:iter);
relchange = relchange(1:iter);

end