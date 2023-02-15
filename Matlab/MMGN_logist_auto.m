function [U, V, rhat, err, outs] = MMGN_logist_auto(Y, omega, M0, sigma, opts)
% This function implements MMGN for 1-bit matrix completion under the logistic
% noise model with a data driven approach for selecting the rank constraint
% parameter r

% --INPUTS-----------------------------------------------------------------------
% Y: observed binary data matrix
% omega: the index set of observations (column-major vectorization)
% M0: the true underlying matrix (for performance tracking)
% sigma: the noise level
% opts: An optional struct containing various options that users may choose to set.
%       opts includes the following parameters
        % rSeq: a grid of values for the rank parameter (default: 1 to 10)
        % maxiters: the maximum number of iterations (default: 100)
        % tol: a tolerance for early stopping (default: 1e-3)
        % stopping: which criteron for early stopping
        %          'objective': early stop when the relative change in the
        %                       objective is less than tol (default)
        %          'estimate': early stop when the relative vhange in the 
        %                      estimate is less than tol
        % rate: the percentage of data to be used as the training set (default: 0.8)
        % seed : set seeds to genearte reproduciable outputs (default: 2022)
        % theta: variant parameter for GNMR, (default: -1, which is the updating variant)

        
        
% --OUTPUTS-----------------------------------------------------------------------
% U: Mhat = U*V'
% V: Mhat = U*V'
% rhat: optimal rank r
% err: norm(Mhat-M0,'fro')^2/norm(M0,'fro')^2 for the final Mhat
% outs: An output structure containing the following 
    % errSeq: the sequence of norm(Mhat-M0,'fro')^2/norm(M0,'fro')^2 when 
    %         estimating the final Mhat with the selected r;
    % relchange: the sequence of the ralative change in the estimate/objective
    %         when estimating the final Mhat with the selected r;
    % obj: the sequence of objective values at each iteration when 
    %         estimating the final Mhat with the selected r;
    
    % Mcell: Estimated M matrix (Mhat) for each r
    % CVrelerr: relative error of Mhat w.r.t. M0 for each r 
    % CVloglik: log-likelihood on the testing set for each r

    
% Xiaoqian Liu
% Dec. 2022

%% Set algorithm parameters from input or by using defaults.
    % Make sure 'opts' struct exists, and is filled with all the options
    if ~exist('opts','var')
        opts = [];
    end
    opts = setDefaults(opts);
    
    rSeq = opts.rSeq;
    numR = length(rSeq);
    maxiters=  opts.maxiters;
    tol = opts.tol;
    stopping = opts.stopping;
    seed  = opts.seed;
    rate = opts.rate;
    theta = opts.theta;
   
    %%
    % Logistic model
    f       = @(x) (1 ./ (1 + exp(-x/sigma)));
    fprime  = @(x) ((1/sigma)*exp(x/sigma) ./ (1 + exp(x/sigma)).^2);
    
%%  Set up the outs structure
    outs = [];
    outs.errSeq = nan(maxiters, 1); 
    outs.relchange = nan(maxiters, 1); 
    outs.obj = nan(maxiters, 1); 
    
    outs.Mcell = cell(numR, 1);
    outs.CVrelerr = nan(numR, 1);
    outs.CVloglik = nan(numR, 1);
    
%% Generate the training data set 
   rng(seed);
   training = randsample(omega, ceil(rate*length(omega)));
   testing = setdiff(omega, training);

   
%% main loop to select r
    for k = 1:numR
        % set the estimated r in this loop
        r = rSeq(k); 

        [U, V, relerr] = MMGN_logist(Y, training, r, M0, sigma,'maxiters',maxiters, 'tol',tol, 'stopping',stopping ,'theta',theta);
        
        Mhat = U*V';
        outs.Mcell{k,1} = Mhat;
        outs.CVrelerr(k, 1) = relerr(end); % this is the rel error on the whole matrix
        
          % compute the log-likelihood on the testing set
%         mhat = Mhat(:);
%         % predict y
%         yhat = sign(f(mhat)-.5); 
%         % compute the error
%         err = ytest-yhat(testing); 
%         outs.CVpderr(k, 1) = length(find(err~=0))/length(err);
        %D = (1+Y)/2;
        outs.CVloglik(k, 1) = -loss_1bit((1+Y)/2, Mhat, testing, f);
    end

    [~, id] = max(outs.CVloglik);
    rhat = rSeq(id);


    %% Now use rhat to reestimate the model
    [U, V, relerr, relchange, obj] = MMGN_logist(Y, omega, rhat, M0, sigma,...
                                                 'maxiters',maxiters, 'tol',tol, 'stopping',stopping, 'theta',theta);
    
    % save the outputs
    err = relerr(end);
    outs.errSeq = relerr; 
    outs.relchange = relchange; 
    outs.obj = obj; 
end
%%

function opts = setDefaults(opts)

        %  rSeq
        if ~isfield(opts,'rSeq')
            opts.rSeq = 1:10;
        end
         %  maxiters
        if ~isfield(opts,'maxiters')
            opts.maxiters = 100;
        end
        
        %  tol
        if ~isfield(opts,'tol')
            opts.tol = 1e-3;
        end
  
        %  stopping
        if ~isfield(opts,'stopping')
            opts.stopping = 'objective';
        end
        
         % seed     
        if ~isfield(opts,'seed')
            opts.seed = 2022;
        end
        
         % rate     
        if ~isfield(opts,'rate')
            opts.rate = 0.8;
        end
        
         % theta--- GNMR variant parameter; updating variant by default.
        if ~isfield(opts,'theta')
            opts.theta = -1;
        end
end
