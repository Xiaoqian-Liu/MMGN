function [U, V, out] = LBFGS_probit(Y, ind_omega, sigma, R, U0, V0, varargin)
% This function implements LBFGS for 1-bit matrix completion under the
% probit noise model, given a specific rank of R.

% --INPUTS-----------------------------------------------------------------------
% Y: the observed binary data matrix, unobserved entries are coded as zero 
% ind_omega: the indicator vector of observations (column-major vectorization)
% sigma: the noise level, assumed to be known
% R: the target rank in the rank constraint
% U0: the initial value of the factor matrix U
% V0: the initial value of the factor matrix V
% varargin: additional parameters for BFGS, including the following
%       m      limited memory parameter (default: 5)
%       tol    relative function value change stopping tolerance {1e-6}, 
%              i.e., the method stops when the relative change of the function value 
%              from one iteration to the next is less than tol.
%       maxiters         maximum number of iterations (default: 1000)
%       DisplayIters     number of iterations to display printed output (default: 100)

% --OUTPUTS-----------------------------------------------------------------------
% U: the factor matrix U, m-by-R, Mhat = U*V'
% V: the factor matrix V, n-by-R, Mhat = U*V'
% out: output of LBFGS

% Xiaoqian Liu
% Dec. 2023

%% Set algorithm parameters from input or by using defaults.
params = inputParser;
params.addParameter('m', 5,  @(x) isscalar(x) & x < 20);
params.addParameter('tol', 1e-6, @isscalar);
params.addParameter('maxiters', 1e3, @(x) isscalar(x) & x > 0);
params.addParameter('DisplayIters', 100, @(x) isscalar(x) & x > 0);
params.parse(varargin{:});

%% Copy from params object.
% tol = params.Results.tol;
% maxiters = params.Results.maxiters;
opts = struct();
opts.M = params.Results.m;
opts.RelFuncTol = params.Results.tol;
%opts.StopTol = params.Results.tol;  %use the default value 1e-5
opts.MaxIters = params.Results.maxiters; 
opts.DisplayIters = params.Results.DisplayIters;

%%
[m, n] = size(Y);
% initialization
x0 = [U0(:); V0(:)];


% Probit model
f      = @(x) normcdf(x,0,sigma);
fprime = @(x) normpdf(x,0,sigma);


%% Implementation
out  = lbfgs( @(x)func_1bit(x, Y, f, fprime, ind_omega),x0, opts);
x_min = out.X;
%
U = reshape(x_min(1:m*R), [m, R]);
V = reshape(x_min((m*R+1):end), [n, R]);
end