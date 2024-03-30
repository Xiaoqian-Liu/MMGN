%Add paths.
addpath('/MMGN'); %load function for MMGN
addpath('/TraceNorm'); %load function for TraceNorm
addpath('/MaxNorm'); %load function for MaxNorm


%%
rng(2022);

d1 = 1000;
d2 = 1000;
r = 1;
sigma = 2;

% Probit model
f      = @(x) normcdf(x,0,sigma);
fprime = @(x) normpdf(x,0,sigma);


%%    
rng(2024) 
% Create underlying matrix
t = 10; % change to 5
s = 30;
while(s>20)
    U0 = trnd(t, [d1,r]);
    V0 = trnd(t, [d2,r]);
    M0 = U0*V0';
    s = sqrt(d1*d2)*max(abs(M0(:)))/norm(M0, 'fro');
end

% generate binary matrix Y
Y0 = sign(f(M0)-rand(d1,d2));
        
% generate the observations
rho = 0.8;
omega = randsample(d1*d2, floor(rho*d1*d2));
ind_omega = zeros(d1*d2, 1);
ind_omega(omega) = 1;

% code unobserved entries as zero
y = Y0(:);
y(ind_omega==0) = 0; % code the unobserved entries as zero
Y = reshape(y, [d1, d2]);
[UU,S,VV] = svd(Y);
U0 = UU(:, 1:r)*sqrt(S(1:r, 1:r));
V0 = VV(:, 1:r)*sqrt(S(1:r, 1:r));

%% MMGN method
opts = [];
opts.rSeq = 1:5;
opts.maxiters = 1e2;
opts.tol = 1e-5;
opts.alpha0 = 3;
opts.solver = 'PCG';
t0 = tic;
[U_mm, V_mm, rhat_mm, relerr_mm, ~] = MMGN_probit_auto(Y, ind_omega, sigma, M0, opts);
t_MMGN = toc(t0);

% save output for MMGN
Mhat_mm = U_mm*V_mm';
relerr_MMGN = relerr_mm;
dist_MMGN = Hellinger_dist(f(Mhat_mm), f(M0));


%% TraceNorm method, use default values for algorithmic parameters

options = struct();
options.iterations = 1e4;
options.stepMax    = 1e9;
options.stepMin    = 1e-4;
options.optTol     = 1e-3;
options.verbosity  = 1;
rSeq = 1:5;
rate = 0.8;
seed = 2022;
t0 = tic;
[Mhat_Trace, rhat, err, ~] = TraceNorm_auto(Y, ind_omega, f, fprime, rSeq, rate, seed, M0, options);
time_Trace = toc(t0);


% save outputs for TraceNorm
relerr_Trace = err;
dist_Trace = Hellinger_dist(f(Mhat_Trace), f(M0));

    
%% Maxnorm method
U00 = UU(:, 1:(r+1))*sqrt(S(1:(r+1), 1:(r+1)));
V00 = VV(:, 1:(r+1))*sqrt(S(1:(r+1), 1:(r+1)));

ticMaxnorm = tic;
[Mhat_max,relerr_max, niterations_max, tau_max] = Max_norm(Y, ind_omega, f, fprime, r, max(abs(M0(:))), U0, V0, M0,...
    'maxiters', 1e3, 'tol', 1e-5);
time_Max = toc(ticMaxnorm);
        
% truncated SVD
[U,S,V] = svd(Mhat_max);
Mhat_Max = U(:,1:r)*S(1:r,1:r)*V(:,1:r)';
% save outputs for MaxNorm
relerr_Max = (norm(Mhat_Max-M0,'fro')/norm(M0,'fro'))^2;
dist_Max = Hellinger_dist(f(Mhat_Max), f(M0));


%%
 save profile_check_t10rho8.mat
