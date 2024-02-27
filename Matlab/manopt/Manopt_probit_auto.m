function [Mhat, rhat, err, outs] = Manopt_probit_auto(Y, ind_omega,  sigma, M, opts)
% This function implements Manifold optimization (Manopt) for 1-bit matrix completion
% under the probit noise model with a data-driven approach for selecting the rank
% constraint parameter r.

% --INPUTS-----------------------------------------------------------------------
% Y: observed binary data matrix, unobserved entries are coded as zero
% ind_omega: the indicator vector of observations (column-major vectorization)
% sigma: the noise level, assumed to be known 
% M: the true underlying matrix (for performance tracking)
% opts: An optional struct containing various options that users may choose to set.
%       opts includes the following parameters
        % rSeq: a grid of values for the rank parameter (default: 1 to 5)
        % maxiters: the maximum number of iterations (default: 1000)
        % tol: a tolerance for early stopping (default: 1e-6)
        % verbosity: how much information to output
        % rate: the percentage of data to be used as the training set (default: 0.8)
        % seed : set seed for reproducibility (default: 2022)

        
        
% --OUTPUTS-----------------------------------------------------------------------
% Mhat: the estimated matrix at the selected rank 
% rhat: the optimal rank r
% err: norm(Mhat-M,'fro')^2/norm(M,'fro')^2 for the final Mhat,
%      given the true matrix M (only for simulation)
% outs: An output structure containing the following 
    % Mcell: Estimated M matrix (Mhat) for each r
    % CVrelerr: relative error of Mhat w.r.t. M for each r 
    % CVloglik: log-likelihood on the testing set for each r

    
% Xiaoqian Liu
% Dec. 2023

%% Set algorithm parameters from input or by using defaults.
    % Make sure 'opts' struct exists, and is filled with all the options
    if ~exist('opts','var')
        opts = [];
    end
    opts = setDefaults(opts);
    
    % for manopt
    maxiters = opts.maxiters;
    tol = opts.tol;
    verbosity = opts.verbosity;
    % for selecting r
    rSeq = opts.rSeq;
    numR = length(rSeq);
    seed  = opts.seed;
    rate = opts.rate;
    
    
    %%
    % Logistic model
    f      = @(x) normcdf(x,0,sigma);
    %fprime = @(x) normpdf(x,0,sigma);
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
    [UU,SS,VV] = svd(Y);
    D = (1+Y)/2;
    d = D(ind_test>0);
    
    rhat = 1;
    mll = -inf;
    U00 = UU(:, 1)*sqrt(SS(1, 1));
    V00 = VV(:, 1)*sqrt(SS(1, 1));   
%% main loop to select r
    for k = 1:numR
        % set the estimated r in this loop
        r = rSeq(k); 
        % initialization according to r
        U0 = UU(:, 1:r)*sqrt(SS(1:r, 1:r));
        V0 = VV(:, 1:r)*sqrt(SS(1:r, 1:r));

        [Mhat1, relerr1] = Manopt_probit(Y, ind_train, sigma, r, U0, V0, M, 'maxiters',maxiters, 'tol',tol, 'verbosity', verbosity);
        
        outs.Mcell{k,1} = Mhat1;
        outs.CVrelerr(k, 1) = relerr1; % this is the rel error on the whole matrix
        
       % compute the log-likelihood on the testing set
        outs.CVloglik(k, 1) = -obj_1bit(d, Mhat1(ind_test>0), f);
        
        % for the overall
        if outs.CVloglik(k, 1)>mll
            rhat = r;
            mll = outs.CVloglik(k, 1);
            U00 = U0;
            V00 = V0;
        end
    end

    %% Now use rhat to reestimate the model
    [Mhat, err] = Manopt_probit(Y, ind_omega, sigma, rhat, U00, V00, M, 'maxiters',maxiters, 'tol',tol, 'verbosity', verbosity);
    
end
%%

function opts = setDefaults(opts)

        %  rSeq
        if ~isfield(opts,'rSeq')
            opts.rSeq = 1:5;
        end
         %  maxiters
        if ~isfield(opts,'maxiters')
            opts.maxiters = 1000;
        end
        
        %  tol
        if ~isfield(opts,'tol')
            opts.tol = 1e-6;
        end
  
         %  tol
        if ~isfield(opts,'verbosity')
            opts.verbosity = 1;
        end
        
         % seed     
        if ~isfield(opts,'seed')
            opts.seed = 2022;
        end
        
         % rate     
        if ~isfield(opts,'rate')
            opts.rate = 0.8;
        end

end
