function [U, V, relerr, relchange, obj, alphas,...
    nBacktracks] = MMGN_logist(Y, omega, R, M0, sigma, varargin)
% This function implements MMGN for 1-bit matrix completion under the logistic
% noise model, given a specific rank R

% --INPUTS-----------------------------------------------------------------------
% Y: observed binary data matrix
% omega: the index set of observations (column-major vectorization)
% R: target rank in the rank constraint
% M0: the true underlying matrix (for performance tracking)
% sigma: the noise level
% varargin: additional parameters, including the following
%       tol: the tolerance for the relative change in the estimate or
%            objective value (for early stopping)
%       maxiters: the maximum number of iterations
%       stopping: which criterion for early stopping
%                 'objective': early stop when the relative change in the
%                              objective is less than tol (default)
%                 'estimate': early stop when the relative vhange in the 
%                             estimate is less than tol
%       theta: variants parameter for GNMR, -1 by default (updating variant)

% --OUTPUTS-----------------------------------------------------------------------
% U: Mhat = U*V'
% V: Mhat = U*V'
% relerr: the sequence of the relative error, defined as
%            norm(Mhat-M0,'fro')^2/norm(M0,'fro')^2
% relchange: the sequence of the ralative change in the estimate/objective
%            for early stopping
% obj: the sequence of objective values at each iteration
% alphas: the stepsize for GN step
% nBacktracks: number of backtracking at each iteration

% Xiaoqian Liu
% Nov. 2022

%% Set algorithm parameters from input or by using defaults.
params = inputParser;
params.addParameter('tol', 1e-3, @isscalar);
params.addParameter('maxiters', 1e2, @(x) isscalar(x) & x > 0);
params.addParameter('stopping', 'objective', @(x) ischar(x)||isstring(x));
params.addParameter('theta', -1, @isscalar); %  we use the updating variant of GNMR by default.
params.parse(varargin{:});

%% Copy from params object.
tol = params.Results.tol;
maxiters = params.Results.maxiters;
stopping = params.Results.stopping;
theta = params.Results.theta;

[m, n] = size(Y);
D = (1+Y)/2;  % the D matrix in the loss function, see equation (4) in the paper


% Logistic model, only need CDF
f       = @(x) (1 ./ (1 + exp(-x/sigma)));


%% Initialization
rng(2022);
U = 0.1*randn(m, R);
V = 0.1*randn(n, R);
Mhat_last = U*V';
obj_last = loss_1bit(D, Mhat_last, omega, f);
% parameters for backtracking
beta = 0.5;
gamma = 0.9;

% allocate memeroy for outputs
obj = nan(maxiters, 1);
relerr = nan(maxiters, 1);
relchange = nan(maxiters, 1);
dec = nan(maxiters, 1);  %decrease in the objective value
alphas = nan(maxiters, 1); % stepsize
nBacktracks = nan(maxiters, 1);  % number of backtracking iteration



L = 1/(4*sigma^2);

for iter = 1:maxiters
    %=======line 3:
    %=======construct X
    X = (4*sigma)*Y.*f(-Y.*Mhat_last); % see corollary 2.1
    
    %======line 4:
    %====== construct x in the LS problem (25)
    x = X(:);
    x = x(omega);
    %====== construct J in the LS problem (25)
    Phi = [kron(V, speye(m)), kron(speye(n), U)];
    Phi = Phi(omega,:);
    
    %======line 5:
    %====== solve the LS problem
    [eta, flag] = lsqr(Phi, x, 1e-10, 1e3);  %change tol from 1e-15 to 1e-10 to save some time
    if flag~=0
        disp('LSQR is unsuccessful!');
    end
    
    %===== get du and dv
    du = eta(1:m*R);
    dv = eta(m*R+1:R*(m+n));
    
    %======line 6:
    %====== recover dU and dV
    dU = reshape(du, [m R]);
    dV = reshape(dv, [R n])';
    
    %====line 7:
    %====update U and V with stepsize alpha chosen by backtracking
    alpha = 1;
    param = (1-theta)/2;
    U1 = param*U + alpha*dU;
    V1 = param*V + alpha*dV;
    
    %=== compute dUV+UdV in the backtracking condition
    G = dU*V' + U*dV';
    g = G(:);
    g = g(omega);
    %===
    dec(iter) = -L*sum(x.*g);  % see eqn(B.9) and the defination of x
    
    %=== start backtracking
    nBacktrack = 0;
    Mhat = U1*V1';
    obj_new = loss_1bit(D, Mhat, omega, f);
    while ( obj_new > obj_last + beta*alpha*dec(iter)) && (nBacktrack <=100)
        alpha = gamma*alpha;
        U1 = param*U + alpha*dU;
        V1 = param*V + alpha*dV;
        Mhat = U1*V1';
        obj_new = loss_1bit(D, Mhat, omega, f);
        nBacktrack = nBacktrack + 1;
    end
    
    %=== save some output
    obj(iter) = obj_new; % the objective value/ loss
    alphas(iter) = alpha;  % stepsize
    nBacktracks(iter) = nBacktrack;  % number of backtracking iteration
    
    % squared relative error
    relerr(iter) = norm(Mhat-M0,'fro')^2/norm(M0,'fro')^2;
    
    if iter>10  %% only early stop when iter>10
        % relative change according to stopping criteron
        if strcmp(stopping,'objective')
            relchange(iter) = abs( (obj_last-obj_new) / (obj_last+eps) );
        else
            relchange(iter) = norm(Mhat-Mhat_last,'fro')^2/norm(Mhat_last,'fro')^2;
        end
        
        %=== early stooping
        if relchange(iter) < tol
            U = U1; % output
            V = V1;
            break;
        end
    end
    
    %=== update for the next iteration
    U = U1;
    V = V1;
    Mhat_last = Mhat;
    obj_last = obj(iter);
end


% outputs
obj = obj(1:iter);
relerr = relerr(1:iter);
relchange = relchange(1:iter);
alphas = alphas(1:iter);
nBacktracks = nBacktracks(1:iter);
end