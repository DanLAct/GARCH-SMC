%%% Performing one-step ahead prediction for the GJR-GARCH model using SMC 
% NOTE: run each function separately

%% Predicting with real data 
% The data required to be loaded below are not available here, but you can use the outputs from 
% this function '[theta, theta_particle, loglike, logprior, W, log_evidence] = GJR_SMC_RW_DataAnneal_parallel(N);' with real data 'Data.mat'.
% After running the above function, then do the following commands:
load('theta.mat');
load('theta_particle.mat');
load('W.mat');
N = length(theta); % Size of population of particles
[h_predict, CI_low_h_t, CI_up_h_t, Predict_median_h_t] = gjr_BayesPostPredict_volatility(N, theta, theta_particle, W);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Predicting with simulated data
% The data 'theta' 'theta_particle' and 'W',which are required to be loaded below, are not available here, but you can use the results
% from this function '[theta, theta_particle, loglike, logprior, W, log_evidence] = GJR_SMC_RW_DataAnneal_parallel(N);' with simulated data 'Data_Sims_gjr.mat'.
% After running the above function, then do the following commands:
load('theta.mat');
load('theta_particle.mat');
load('W.mat');
N = length(theta); % Size of population of particles
load('Sim_Var_gjr.mat');
[h_predict, CI_low_h_t, CI_up_h_t, Predict_median_h_t, ELPD_LFO_true_var] = gjr_BayesPostPredict_volatility_sim(N, Sim_Var, theta, theta_particle, W);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

