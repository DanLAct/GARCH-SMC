%%% Performing posterior estimation for the BEGE model of real data using MCMC and SMC methods
% NOTE: each function will take many hours to run; run each function separately

%% MCMC with 'biased likelihood'
load('start_smc_real.mat');
load('cov_smc_real.mat');
N = 1000000; % can change the value of N as desired
h = 0.1;
[theta, theta_transf, loglike, logprior] = MCMC_biased(N,start_smc,cov_smc,h);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MCMC with 'unbiased likelihood'
load('start_smc_unbiased_real.mat');
load('cov_smc_unbiased_real.mat');
N = 1000000; % can change the value of N as desired
h = 0.1;
[theta, theta_transf, loglike, logprior] = MCMC_unbiased_is(N,start_smc,cov_smc,h);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Data annealing SMC with 'biased likelihood'
N = 10000; % can change the value of N as desired
[theta, theta_particle, loglike, logprior, W, log_evidence] = SMC_RW_DataAnneal_parallel_biased(N);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Data annealing SMC with 'unbiased likelihood'(The likelihood is unbiasedly estimated using Monte Carlo Integration)
N = 10000; % can change the value of N as desired
[theta, theta_particles, loglike, logprior, W, log_evidence] = SMC_RW_DataAnneal_Par_unif_unbiased_mc(N);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Data annealing SMC with 'unbiased likelihood'(The likelihood is unbiasedly estimated using importance sampling)
N = 10000; % can change the value of N as desired
[theta, theta_particles, loglike, logprior, W, log_evidence] = SMC_RW_DataAnneal_Par_unif_unbiased_is(N);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Likelihood annealing SMC with 'biased likelihood'
N = 10000; % can change the value of N as desired
[theta, theta_particle, loglike, logprior, gamma, log_evidence] = SMC_RW_LikeAnneal_parallel_biased(N);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Likelihood annealing SMC with 'unbiased likelihood'(The likelihood is unbiasedly estimated using importance sampling)
N = 10000; % can change the value of N as desired
[theta, theta_particle, loglike, logprior, gamma, log_evidence] = SMC_RW_LikeAnneal_ParVec_unbiased_is(N);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




