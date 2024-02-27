function [Mhat, relerr] = Manopt_probit(Y, ind_omega, sigma, R, U0, V0, M, varargin)
% This function implements Manifold optimization (Manopt) for 1-bit matrix 
% completion under the probit noise model, given a specific rank of R.

% --INPUTS-----------------------------------------------------------------------
% Y: the observed binary data matrix, unobserved entries are coded as zero 
% ind_omega: the indicator vector of observations (column-major vectorization)
% sigma: the noise level, assumed to be known
% R: the target rank 
% U0: the initial value of the factor matrix U
% V0: the initial value of the factor matrix V
% M: the true underlying matrix (for performance tracking)
% varargin: additional parameters, including the following
%       tol: the tolerance for the relative change in the objective value or
%            norm of the estimate (for early stopping), default is 1e-6.
%       maxiters: the maximum number of iterations, default is 1000.
%       verbosity: how much information to output.
%
% --OUTPUTS-----------------------------------------------------------------------
% Mhat: the estimated matrix
% relerr: the relative error at the final estimate
%
% Xiaoqian Liu
% Dec. 2023    

%% Set algorithm parameters from input or by using defaults.
params = inputParser;
params.addParameter('tol', 1e-6, @isscalar); % same default value in trust region
params.addParameter('maxiters', 1e3, @(x) isscalar(x) & x > 0); % same default value in trust region
params.addParameter('verbosity', 1, @isscalar); % how much information to output
params.parse(varargin{:});

%% Copy from params object.
tol = params.Results.tol;
maxiters = params.Results.maxiters;
verbosity = params.Results.verbosity;

%%
[m, n] = size(Y);
y_vec = Y(ind_omega>0); % vector of binary observations 
d_vec = (1+y_vec)/2;
    
f      = @(x) normcdf(x,0,sigma);
fprime = @(x) normpdf(x,0,sigma);

%% Initialization
[U, S, V] = svds(U0*V0', R);
X0.U = U;
X0.S = S;
X0.V = V;

%% set elements for manifold optimization

% Pick the manifold of matrices of size mxn of fixed rank r.
problem.M = fixedrankembeddedfactory(m, n, R);
problem.cost = @cost;
problem.egrad = @egrad; 

% Define the problem cost function.
function fv = cost(M)

    MM = M.U*M.S*M.V';
    m_vec = MM(ind_omega>0);

    m1 = m_vec(d_vec>0);
    m0 = m_vec(d_vec==0);
    fv = -sum(log(f(m1))) - sum(log(1 - f(m0)));

end

% Define the Euclidean gradient of the cost function, that is, the
% gradient of f(X) as a standard function of X.
function G = egrad(M)

    XX = M.U*M.S*M.V';
    m_vec = XX(ind_omega>0);

    g = -y_vec.*fprime(m_vec)./f(y_vec.*m_vec);

    G = zeros(m, n);
    G(ind_omega>0) = g;
    G = sparse(G);
end


% Minimize the cost function using Riemannian trust-regions, starting
% from the initial guess X0.
options.tolgradnorm = tol;
options.maxiter = maxiters;
options.verbosity = verbosity;
%[X, Xcost, info]  = trustregions(problem, X0, options);
%[X, Xcost, info] = steepestdescent(problem, X0);
%[X, Xcost, info] = rlbfgs(problem, X0, options);
[X, Xcost, info] = barzilaiborwein(problem, X0, options);

% The reconstructed matrix is X, represented as a structure with fields
% U, S and V.
Mhat = X.U*X.S*X.V';
relerr = norm(Mhat-M,'fro')^2/norm(M,'fro')^2;

end