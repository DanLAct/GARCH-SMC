%%% Performing posterior estimation for the GARCH model of real data using SMC
% NOTE: run each function separately

%% Data annealing SMC 
N = 10000; % can change the value of N as desired
[theta, theta_particle, loglike, logprior, W, log_evidence] = garch_SMC_RW_DataAnneal_parallel(N);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Likelihood annealing SMC 
N = 10000; % can change the value of N as desired
[theta, theta_particle, loglike, logprior, gamma, log_evidence] = garch_SMC_RW_LikeAnneal_parallel(N);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
