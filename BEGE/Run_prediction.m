%%% Performing one-step ahead prediction for the BEGE model using MCMC and SMC methods
% NOTE: each function will take many hours to run; run each function separately

%% Predicting with real data (The likelihood is unbiasedly estimated using importance sampling)
% The data required to be loaded below are not available here, but you can use the results
% from this function '[theta, theta_particles, loglike, logprior, W,
% log_evidence] = SMC_RW_DataAnneal_Par_unif_unbiased_is(N);' with real data 'Data.mat'.
% After running the above function, rename the output 'theta_particles' to 'theta_particle', and then do the following commands:
load('theta.mat');
load('theta_particle.mat');
load('W.mat');
N = length(theta); % Size of population of particles
[h_predict, CI_low_h_t, CI_up_h_t, Predict_median_h_t] = bege_BayesPostPredict_volatility_unbiased_is(N, theta, theta_particle, W);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Predicting with simulated data (The likelihood is unbiasedly estimated using importance sampling)
% The data 'theta' 'theta_particle' and 'W',which are required to be loaded below, are not available here, but you can use the results
% from this function '[theta, theta_particles, loglike, logprior, W,
% log_evidence] = SMC_RW_DataAnneal_Par_unif_unbiased_is(N);' with simulated data 'Data_Sims.mat'.
% After running the above function, rename the output 'theta_particles' to 'theta_particle', and then do the following commands:
load('theta.mat');
load('theta_particle.mat');
load('W.mat');
N = length(theta); % Size of population of particles
load('Sim_Var.mat');
[h_predict, CI_low_h_t, CI_up_h_t, Predict_median_h_t, ELPD_LFO_true_var] = bege_BayesPostPredict_volatility_sim_unbiased(N, Sim_Var, theta, theta_particle, W);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%