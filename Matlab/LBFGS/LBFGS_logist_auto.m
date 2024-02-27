function [U, V, rhat, err, out, outs] = LBFGS_logist_auto(Y, ind_omega, sigma, M, opts)
% This function implements LBFGS for 1-bit matrix completion under the
% logistic noise model with a data-driven approach for selecting the rank constraint
% parameter r.

% --INPUTS-----------------------------------------------------------------------
% Y: observed binary data matrix, unobserved entries are coded as zero
% ind_omega: the indicator vector of observations (column-major vectorization)
% sigma: the noise level, assumed to be known
% M0: the true underlying matrix (for performance tracking) 
% opts: An optional struct containing various options that users may choose to set.
%       opts includes the following parameters
        % rSeq: a grid of values for the rank parameter (default: 1 to 5)
        % maxiters: the maximum number of iterations (default: 1e3)
        % tol: a tolerance for early stopping (default: 1e-6)
        % rate: the percentage of data to be used as the training set (default: 0.8)
        % seed : set seed to genearte reproduciable outputs (default: 2022)
        % m: limited memory parameter (default: 5)
        % DisplayIters: number of iterations to display printed output (default: 100)
        
        
% --OUTPUTS-----------------------------------------------------------------------
% U: the factor matrix U, Mhat = U*V'
% V: the factor matrix V, Mhat = U*V'
% rhat: the optimal rank r, equal to the number of columns of U and V
% err: estimation error of Mhat, equal to norm(Mhat-M,'fro')^2/norm(M,'fro')^2 
% out: output of LBFGS at rhat
% outs: outputs of LBFGS in the process of selecting r
    
% Xiaoqian Liu
% Dec. 2023

%% Set algorithm parameters from input or by using defaults.
    % Make sure 'opts' struct exists, and is filled with all the options
    if ~exist('opts','var')
        opts = [];
    end
    opts = setDefaults(opts);
    
    % for LBFGS
    tol = opts.tol;
    maxiters=  opts.maxiters;
    mm = opts.m; % to avoid overuse of notation
    DisplayIters = opts.DisplayIters;
    % for selecting r
    rSeq = opts.rSeq;
    numR = length(rSeq);
    seed  = opts.seed;
    rate = opts.rate;
     %%
    % Logistic model
    f       = @(x) (1 ./ (1 + exp(-x/sigma)));
    %fprime  = @(x) ((1/sigma)*exp(x/sigma) ./ (1 + exp(x/sigma)).^2);
    [m, n] = size(Y);

%%  Set up the outs structure
    outs = [];
    
    outs.Mcell = cell(numR, 1);
    outs.CVrelerr = nan(numR, 1);
    outs.CVloglik = nan(numR, 1);
    
%% Generate the training data set 
   rng(seed);
   omega = find(ind_omega);
   loc_train = randsample(omega, ceil(rate*length(omega)));
   loc_test =  setdiff(omega, loc_train);
   ind_train = zeros(m*n, 1);
   ind_train(loc_train)=1;
   ind_test = zeros(m*n, 1);
   ind_test(loc_test)=1;

 %% for initialization
   [UU,S,VV] = svd(Y);
   D = (1+Y)/2;
   d = D(ind_test>0);
   
   
   rhat = 1;
   mll = -inf;
   U00 = UU(:, 1)*sqrt(S(1, 1));
   V00 = VV(:, 1)*sqrt(S(1, 1));  
%% main loop to select r
    for k = 1:numR
        % set the estimated r in this loop
        r = rSeq(k);
        % initialization according to r
        U0 = UU(:, 1:r)*sqrt(S(1:r, 1:r));
        V0 = VV(:, 1:r)*sqrt(S(1:r, 1:r));
        
        [U, V, ~] = LBFGS_logist(Y, ind_train, sigma, r, U0, V0, 'maxiters',maxiters, 'tol',tol,...
                                                          'm', mm, 'DisplayIters', DisplayIters);
        
        
        Mhat = U*V';
        err = norm(Mhat - M, 'fro')^2/norm(M, 'fro')^2;
        outs.Mcell{k,1} = Mhat;
        outs.CVrelerr(k, 1) = err; % this is the rel error on the whole matrix
        
        % compute the log-likelihood on the testing set
        outs.CVloglik(k, 1) = -obj_1bit(d, Mhat(ind_test>0), f);
        % for the overall
        if outs.CVloglik(k, 1)>mll
            rhat = r;
            mll = outs.CVloglik(k, 1);
            U00 = U;
            V00 = V;
        end
    end
    %% Now use rhat to reestimate the model
    [U, V, out] = LBFGS_logist(Y, ind_omega, sigma, rhat, U00, V00,'maxiters',maxiters, 'tol',tol,...
                                                   'm', mm, 'DisplayIters', DisplayIters);
    
    % save the outputs
    err = norm(U*V' - M, 'fro')^2/norm(M, 'fro')^2;
end
%%

function opts = setDefaults(opts)

        %  rSeq
        if ~isfield(opts,'rSeq')
            opts.rSeq = 1:5;
        end
         %  maxiters
        if ~isfield(opts,'maxiters')
            opts.maxiters = 1e3;
        end
        
        %  tol
        if ~isfield(opts,'tol')
            opts.tol = 1e-6;
        end
  
        %  m
        if ~isfield(opts,'m')
            opts.m = 5;
        end
        
         % seed     
        if ~isfield(opts,'seed')
            opts.seed = 2022;
        end
        
         % rate     
        if ~isfield(opts,'rate')
            opts.rate = 0.8;
        end
        
        
        
        % DisplayIters
        if ~isfield(opts,'DisplayIters')
            opts.DisplayIters = 100;
        end
end
