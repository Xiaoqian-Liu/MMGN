function [U, V, relerr, relchange, obj, alphas, nBacktracks] = GD_probit(Y, ind_omega, sigma, U0, V0, M, varargin)
% This function implements GD for 1-bit matrix completion under the probit 
% noise model, given a specific rank of R (specified through the number of 
% columns of U0 and V0).

% --INPUTS-----------------------------------------------------------------------
% Y: the observed binary data matrix, unobserved entries are coded as zero 
% ind_omega: the indicator vector of observations (column-major vectorization)
% sigma: the noise level, assumed to be known
% U0: the initial value of the factor matrix U
% V0: the initial value of the factor matrix V
% M: the true underlying matrix (for performance tracking)
% varargin: additional parameters, including the following
%       tol: the tolerance for the relative change in the  objective value or
%            norm of the estimate. It is provided for early stopping with
%            a default value of 1e-4.
%       maxiters: the maximum number of iterations, the default is 100.
%       stopping: the criterion used for early stopping
%                 'objective': early stop the algorithm when the relative change 
%                             in the objective value is less than tol (default)
%                 'estimate': early stop the algorithm when the relative change 
%                             in the estimate (squared F-norm) is less than tol
%       alpha0: the initial stepsize for backtracking linesearch (default is 1)


% --OUTPUTS-----------------------------------------------------------------------
% U: the factor matrix U, m-by-R, Mhat = U*V'
% V: the factor matrix V, n-by-R, Mhat = U*V'
% relerr: the sequence of the relative error, defined as 
%            norm(Mhat-M,'fro')^2/norm(M,'fro')^2,
%         given the true matrix M (only for simulation)
% relchange: the sequence of the ralative change in the objective/estimate
%            (based on the stoping criterion)
% obj: the sequence of the objective value at each iteration
% alphas: the sequence of the stepsize for the GD step at each iteration
% nBacktracks: the sequence of the number of backtracking at each iteration

% Xiaoqian Liu
% Dec. 2023

%% Set algorithm parameters from input or by using defaults.
params = inputParser;
params.addParameter('tol', 1e-4, @isscalar);
params.addParameter('maxiters', 1e2, @(x) isscalar(x) & x > 0);
params.addParameter('stopping', 'objective', @(x) ischar(x)||isstring(x));
params.addParameter('alpha0', 1, @isscalar); %  the initial stepsize of backtracking
params.parse(varargin{:});

%% Copy from params object.
tol = params.Results.tol;
maxiters = params.Results.maxiters;
stopping = params.Results.stopping;
alpha0 = params.Results.alpha0;

%%
[m, n] = size(Y);
y = Y(ind_omega>0); % vector of binary observations 
d = (1+y)/2;

f      = @(x) normcdf(x,0,sigma);
fprime = @(x) normpdf(x,0,sigma);

%% Initialization
Mhat_last = U0*V0';
mhat_last = Mhat_last(ind_omega>0); 
obj_last = obj_1bit(d, mhat_last, f);
% parameters for backtracking
beta = 0.5;
gamma = 0.9;

%% allocate memeroy for outputs
obj = nan(maxiters, 1);
relerr = nan(maxiters, 1);
relchange = nan(maxiters, 1);
dec = nan(maxiters, 1);  %decrease in the objective value
alphas = nan(maxiters, 1); % stepsize
nBacktracks = nan(maxiters, 1);  % number of backtracking iteration

%%
for iter = 1:maxiters
    
    %=======compute gradient
    x = -y.*fprime(mhat_last)./f(y.*mhat_last);
    X = zeros(m,n);
    X(ind_omega>0) = x;
    X = sparse(X);

    A1 = X*V0;
    A2 = X'*U0;
 
   
    % update iterate
    alpha = alpha0;   
    U = U0 - alpha* A1;
    V = V0 - alpha* A2;
    Mhat = U*V';
    mhat = Mhat(ind_omega>0);
    obj_new = obj_1bit(d, mhat, f);

    %=== backtracking
    dec(iter) = norm(A1, 'fro')^2 + norm(A2, 'fro')^2 ;
    nBacktrack = 0;
    
    while (obj_new > obj_last-beta*alpha*dec(iter))  && (nBacktrack <=100)
        alpha = gamma*alpha;
        U = U0 - alpha* A1;
        V = V0 - alpha* A2;
        Mhat = U*V';
        mhat = Mhat(ind_omega>0);
        obj_new = obj_1bit(d, mhat, f);
        nBacktrack = nBacktrack + 1;
    end
    
    %=== save some output
    obj(iter) = obj_new; % the objective value/ loss
    alphas(iter) = alpha;  % stepsize
    nBacktracks(iter) = nBacktrack;  % number of backtracking iteration
    
    % squared relative error
    relerr(iter) = norm(Mhat-M,'fro')^2/norm(M,'fro')^2;
    
    if iter>10  %% only early stop when iter>10
        % relative change according to stopping criteron
        if strcmp(stopping,'objective')
            relchange(iter) = abs( (obj_last-obj_new) / (obj_last+eps) );
        else
            relchange(iter) = norm(Mhat-Mhat_last,'fro')^2/norm(Mhat_last,'fro')^2;
        end
        %=== early stooping
        if relchange(iter) < tol
            break;
        end
    end
    
    %=== update for the next iteration
    U0 = U;
    V0 = V;
    mhat_last = Mhat(ind_omega>0);
    obj_last = obj(iter);
end

% outputs
obj = obj(1:iter);
relerr = relerr(1:iter);
relchange = relchange(1:iter);
alphas = alphas(1:iter);
nBacktracks = nBacktracks(1:iter);
end