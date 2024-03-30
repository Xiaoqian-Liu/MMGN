%% load the raw results
clear;
load('profile_check_t10rho8.mat')
%%
m0 = M0(:);
d0 = f(m0);
%% Compute relative error in each value group
err_trace = zeros(1, 5);
err_max = zeros(1, 5);
err_mmgn = zeros(1, 5);


%  value group 1
m1 = m0(m0<=-2.5);
indset = find(m0<=-2.5);

m_trace =  Mhat_Trace(:);
m_trace = m_trace(indset);
err_trace(1, 1) = norm(m1- m_trace, 'fro')^2/(norm(m0, 'fro')^2);

m_max =  Mhat_Max(:);
m_max = m_max(indset);
err_max(1, 1) = norm(m1- m_max, 'fro')^2/(norm(m0, 'fro')^2);

m_mmgn =  Mhat_mm(:);
m_mmgn = m_mmgn(indset);
err_mmgn(1, 1) = norm(m1- m_mmgn, 'fro')^2/norm(m0, 'fro')^2;
    

% value group 2
m1 = m0(m0> -2.5 & m0<=-0.5);
indset = find(m0> -2.5 & m0<=-0.5);

m_trace =  Mhat_Trace(:);
m_trace = m_trace(indset);
err_trace(1, 2) = norm(m1- m_trace, 'fro')^2/(norm(m0, 'fro')^2);

m_max =  Mhat_Max(:);
m_max = m_max(indset);
err_max(1, 2) = norm(m1- m_max, 'fro')^2/(norm(m0, 'fro')^2);

m_mmgn =  Mhat_mm(:);
m_mmgn = m_mmgn(indset);
err_mmgn(1, 2) = norm(m1- m_mmgn, 'fro')^2/norm(m0, 'fro')^2;


     
% value group 3
m1 = m0(m0> -0.5 & m0<=0.5);
indset = find(m0> -0.5 & m0<=0.5);

m_trace =  Mhat_Trace(:);
m_trace = m_trace(indset);
err_trace(1, 3) = norm(m1- m_trace, 'fro')^2/(norm(m0, 'fro')^2);

m_max =  Mhat_Max(:);
m_max = m_max(indset);
err_max(1, 3) = norm(m1- m_max, 'fro')^2/(norm(m0, 'fro')^2);

m_mmgn =  Mhat_mm(:);
m_mmgn = m_mmgn(indset);
err_mmgn(1, 3) = norm(m1- m_mmgn, 'fro')^2/norm(m0, 'fro')^2;
    


% value group 4
m1 = m0(m0>0.5 & m0<=2.5);
indset = find(m0>0.5 & m0<=2.5);

m_trace =  Mhat_Trace(:);
m_trace = m_trace(indset);
err_trace(1, 4) = norm(m1- m_trace, 'fro')^2/(norm(m0, 'fro')^2);

m_max =  Mhat_Max(:);
m_max = m_max(indset);
err_max(1, 4) = norm(m1- m_max, 'fro')^2/(norm(m0, 'fro')^2);

m_mmgn =  Mhat_mm(:);
m_mmgn = m_mmgn(indset);
err_mmgn(1, 4) = norm(m1- m_mmgn, 'fro')^2/norm(m0, 'fro')^2;
    

    
    
% value group 5
m1 = m0(m0>2.5);
indset = find(m0>2.5);

m_trace =  Mhat_Trace(:);
m_trace = m_trace(indset);
err_trace(1, 5) = norm(m1- m_trace, 'fro')^2/(norm(m0, 'fro')^2);

m_max =  Mhat_Max(:);
m_max = m_max(indset);
err_max(1, 5) = norm(m1- m_max, 'fro')^2/(norm(m0, 'fro')^2);

m_mmgn =  Mhat_mm(:);
m_mmgn = m_mmgn(indset);
err_mmgn(1, 5) = norm(m1- m_mmgn, 'fro')^2/norm(m0, 'fro')^2;
    



%% Compute Hellinger distance in each probability group
dist_trace = zeros(1, 5);
dist_max = zeros(1, 5);
dist_mmgn = zeros(1, 5);


% prob.  group 1
d1 = d0(d0<=f(-2.5));
indset = find(d0<=f(-2.5));

m_trace =  Mhat_Trace(:);
d_trace = f(m_trace);
d_trace = d_trace(indset);
dist_trace(1, 1) = Hellinger_dist(d_trace, d1)*length(d_trace);

m_max =  Mhat_Max(:);
d_max = f(m_max);
d_max = d_max(indset);
dist_max(1, 1) = Hellinger_dist(d_max, d1)*length(d_trace);
    
m_mmgn =  Mhat_mm(:);
d_mmgn = f(m_mmgn);
d_mmgn = d_mmgn(indset);
dist_mmgn(1, 1) = Hellinger_dist(d_mmgn, d1)*length(d_trace);



% prob. group 2
d1 = d0(d0>f(-2.5) & d0<=f(-0.5));
indset = find(d0>f(-2.5) & d0<=f(-0.5));

m_trace =  Mhat_Trace(:);
d_trace = f(m_trace);
d_trace = d_trace(indset);
dist_trace(1, 2) = Hellinger_dist(d_trace, d1)*length(d_trace);

m_max =  Mhat_Max(:);
d_max = f(m_max);
d_max = d_max(indset);
dist_max(1, 2) = Hellinger_dist(d_max, d1)*length(d_trace);

m_mmgn =  Mhat_mm(:);
d_mmgn = f(m_mmgn);
d_mmgn = d_mmgn(indset);
dist_mmgn(1, 2) = Hellinger_dist(d_mmgn, d1)*length(d_trace);
    
    


% prob. group 3
d1 = d0(d0>f(-0.5) & d0<=f(0.5));
indset = find(d0>f(-0.5) & d0<=f(0.5));

m_trace =  Mhat_Trace(:);
d_trace = f(m_trace);
d_trace = d_trace(indset);
dist_trace(1, 3) = Hellinger_dist(d_trace, d1)*length(d_trace);

m_max =  Mhat_Max(:);
d_max = f(m_max);
d_max = d_max(indset);
dist_max(1, 3) = Hellinger_dist(d_max, d1)*length(d_trace);

m_mmgn =  Mhat_mm(:);
d_mmgn = f(m_mmgn);
d_mmgn = d_mmgn(indset);
dist_mmgn(1, 3) = Hellinger_dist(d_mmgn, d1)*length(d_trace);
    

    
% prob. group 4
d1 = d0(d0>f(0.5) & d0<=f(2.5));
indset = find(d0>f(0.5) & d0<=f(2.5));

m_trace =  Mhat_Trace(:);
d_trace = f(m_trace);
d_trace = d_trace(indset);
dist_trace(1, 4) = Hellinger_dist(d_trace, d1)*length(d_trace);

m_max =  Mhat_Max(:);
d_max = f(m_max);
d_max = d_max(indset);
dist_max(1, 4) = Hellinger_dist(d_max, d1)*length(d_trace);

m_mmgn =  Mhat_mm(:);
d_mmgn = f(m_mmgn);
d_mmgn = d_mmgn(indset);
dist_mmgn(1, 4) = Hellinger_dist(d_mmgn, d1)*length(d_trace);



% prob. group 5
d1 = d0(d0>f(2.5));
indset = find(d0>f(2.5));

m_trace =  Mhat_Trace(:);
d_trace = f(m_trace);
d_trace = d_trace(indset);
dist_trace(1, 5) = Hellinger_dist(d_trace, d1)*length(d_trace);

m_max =  Mhat_Max(:);
d_max = f(m_max);
d_max = d_max(indset);
dist_max(1, 5) = Hellinger_dist(d_max, d1)*length(d_trace);

m_mmgn =  Mhat_mm(:);
d_mmgn = f(m_mmgn);
d_mmgn = d_mmgn(indset);
dist_mmgn(1, 5) = Hellinger_dist(d_mmgn, d1)*length(d_trace);
   

%% save the data to make plots
save t10rho8-max5-results.mat err_mmgn dist_mmgn err_trace dist_trace err_max dist_max m0
