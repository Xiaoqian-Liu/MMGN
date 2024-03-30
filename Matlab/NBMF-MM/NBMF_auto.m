function [U, V, rhat, outs] = NBMF_auto(D, omega, opts)
% This function translate the NBMF algorithm in  
% ''A majorization-minimization algorithm for nonnegative binary matrix
% factorization''. Original python code in https://github.com/magronp/NMF-binary.
% It selects the rank in a data-driven manner. 

% --INPUTS-----------------------------------------------------------------------
% D: observed binary data matrix, zero-one
% omega: the index set of observations (column-major vectorization)
% opts: an optional structure containing various options that users may choose to set.
%       opts includes the following parameters
        % rSeq: a grid of values for the rank parameter (default: 1 to 5)
        % maxiters: the maximum number of iterations (default: 1e3)
        % tol: a tolerance for early stopping (default: 1e-6)
        % alpha: prior parameter for H, default is 1
        % beta: prior parameter for H, default is 1
        % rate: the percentage of data to be used as the training set (default: 0.8)
        % seed : set seed for reproducibility (default: 2022)

% --OUTPUTS-----------------------------------------------------------------------
% U: Mhat = U*V, U is nonnegative
% V: Mhat = U*V, V is nonnegative
% rhat: the optimal rank r, equal to the number of columns of U
% outs: An output structure containing the following 
    % relchange: the sequence of the ralative change in the objective
    %         when estimating the final Mhat with the selected r;
    % obj: the sequence of objective values at each iteration when 
    %         estimating the final Mhat with the selected r;
    
    % Mcell: Estimated M matrix (Mhat) for each r
    % CVloss: loss on the testing set for each r


% Xiaoqian Liu
% March. 2024

%% Set algorithm parameters from input or by using defaults.
    % Make sure 'opts' struct exists, and is filled with all the options
if ~exist('opts','var')
        opts = [];
    end
    opts = setDefaults(opts);
    % for MMGN
    maxiters = opts.maxiters;
    tol = opts.tol;
    alpha = opts.alpha;
    beta = opts.beta;
    % for selecting r
    rSeq = opts.rSeq;
    numR = length(rSeq);
    seed  = opts.seed;
    rate = opts.rate;

%%
D(isnan(D)) = 0;
%[m, n] = size(D); % D is zero or one
    %%  Set up the outs structure
    outs = [];
    outs.relchange = nan(maxiters, 1);
    outs.obj = nan(maxiters, 1);
    
    outs.Mcell = cell(numR, 1);
    outs.CVloss = nan(numR, 1);
  %% Generate the training data set
    rng(seed);
    loc_train = randsample(omega, ceil(rate*length(omega)));
    loc_test =  setdiff(omega, loc_train);

    loss_pre = Inf;
 %%
 for k = 1:numR
        % set the estimated r in this loop
        r = rSeq(k);
        [U, V, obj, relchange] = NBMF(D, loc_train, r, 'alpha', alpha, 'beta', beta,...
                                            'maxiters', maxiters, 'tol', tol); 
        Mhat = U*V;
        outs.Mcell{k,1} = Mhat;

        % compute the log-likelihood on the testing set
        outs.CVloss(k, 1) = perplx_NBMF(D, U, V, loc_test);
        
        % for the overall
        if outs.CVloss(k, 1)<loss_pre
            rhat = r;
            loss_pre = outs.CVloss(k, 1);
        end
  end

    %% Now use rhat to reestimate the model
    [U, V, obj, relchange] = NBMF(D, omega, rhat, 'alpha', alpha, 'beta', beta,...
                                            'maxiters', maxiters, 'tol', tol); 
    
    % save the outputs
    outs.obj = obj;
    outs.relchange = relchange;
end



function opts = setDefaults(opts)

        %  rSeq
        if ~isfield(opts,'rSeq')
            opts.rSeq = 1:5;
        end
         %  maxiters
        if ~isfield(opts,'maxiters')
            opts.maxiters = 2e3;
        end
        
        
        %  tol
        if ~isfield(opts,'tol')
            opts.tol = 1e-6;
        end
          
        
         % seed     
        if ~isfield(opts,'seed')
            opts.seed = 2022;
        end
        
         % rate     
        if ~isfield(opts,'rate')
            opts.rate = 0.8;
        end
        
        
        if ~isfield(opts,'alpha')
            opts.alpha = 1;
        end
        
        if ~isfield(opts,'beta')
            opts.beta = 1;
        end
end
