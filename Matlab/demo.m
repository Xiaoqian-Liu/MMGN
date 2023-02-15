%%  set some problem parameters
rng(2022);

m = 1000;
n = 1000;
r = 1;
rho = 0.5;
sigma =0.5 ;


% generate the underlying matrix, non-spiky
U0 = unifrnd(-0.5,0.5,m, r);
V0 = unifrnd(-0.5,0.5,n, r);
M0 = U0*V0';
M0 = M0/max(abs(M0(:)));
sratio = sqrt(m*n)*max(abs(M0(:)))/norm(M0, 'fro');


%% demo 1, probit model
f      = @(x) normcdf(x,0,sigma);
% generate the observed 1-bit matrix Y
Y = sign(f(M0)-rand(m,n));

% generate the index set of observations
omega = randsample(m*n, floor(rho*m*n));


% try MMGN 
% given the rank r, play with 'stopping' (objective or estimate)
t0 = tic;
[Uhat1, Vhat1, relerr1] = MMGN_probit(Y, omega, r, M0, sigma, 'maxiters', 1e2, 'tol', 1e-4, 'stopping', 'objective');
toc(t0)

% data-driven approach to select r  ***** takes about 2 min ****
opts = [];
opts.rSeq = 1:5;
opts.maxiters = 1e2;
opts.tol = 1e-4;
opts.stopping = 'objective'; % play with 'stopping' 
t0 = tic;
[Uhat2, Vhat2, rhat, relerr2, outs] = MMGN_probit_auto(Y, omega, M0, sigma, opts);
toc(t0)


%% demo 2, logistic model
f       = @(x) (1 ./ (1 + exp(-x/sigma)));
% generate the observed 1-bit matrix Y
Y = sign(f(M0)-rand(m,n));
y = Y(:);
% generate the index set of observations
omega = randsample(m*n, floor(rho*m*n));


% try MMGN 
% given the rank r
t0 = tic;
[Uhat1, Vhat1, relerr1] = MMGN_logist(Y, omega, r, M0, sigma, 'maxiters', 1e2, 'tol', 1e-4, 'stopping', 'objective');
toc(t0)

%% data-driven approach to select r  ***** takes about 2.5 min ****
opts = [];
opts.rSeq = 1:5;
opts.maxiters = 1e2;
opts.tol = 1e-4;
opts.stopping = 'objective';% play with 'stopping' and see the difference in run time and estimation error
t0 = tic;
[Uhat2, Vhat2, rhat, relerr2, outs] = MMGN_logist_auto(Y, omega, M0, sigma, opts);
toc(t0)