function [h_predict, CI_low_h_t, CI_up_h_t, Predict_median_h_t, ELPD_LFO_true_var] = garch_BayesPostPredict_volatility_sim(N, Sim_Var, theta, theta_particle, W)
%One-step ahead forecast of conditional variances together with LFO-CV under GARCH(1,1) model for Simulated Data, with using the output of parameters sampled from Data Annealing SMC
%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%
% N                  - Size of population of particles
% Sim_Var            - Conditional variance of the simulated data
% theta              - Parameters sampled from Data Annealing SMC: N samples from different amount of available observations: y_{1:t}, t=1,..,T
% theta_particle     - Transformed theta
% W                  - The weights of weighted samples, corresponding to the above thetas/theta_particles

%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%
% h_predict            - N samples of the one-step ahead forecast of conditional variances
% CI_low_h_t           - 0.025 quantile of the above posterior predictive sample of the conditional variances
% CI_up_h_t            - 0.975 quantile of the above posterior predictive sample of the conditional variances
% Predict_median_h_t   - Median of the above posterior predictive sample of the conditional variances
% ELPD_LFO_true_var    - Approximated epld_LFO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%load('Prediction_TrueGarch_unif_sim.mat');
%load('Sim_Var_garch.mat');

%loading simulated data
load('Data_Sims_garch.mat');
rate_return = Simulation;
rate_return(isnan(rate_return)) = 0;

%Ranges for parameters transformation
mylims = zeros(4,2);
mylims(3,1) = 0.2;
mylims(4,1) = -0.9;
mylims(:,2) = [0.3; 0.5; 0.99; 0.9];

T = length(rate_return); % total number of data
start=200; % Starting time point for making prediction; the time point for the 1st predicted conditional variance is: t = start+1
PT=T-start;

CI_low_h_t=zeros(1,PT);
CI_up_h_t=zeros(1,PT);
Predict_median_h_t=zeros(1,PT);
params=theta;
h_predict = zeros(N,PT);
h_garch = Sim_Var;

parfor t=1:PT
    fprintf('Just starting with the %ith prediction.\n',t);
    
    [~,h] = garch_loglike_parallel(theta_particle(:,:,t+start-1),rate_return(1:t+start-1),mylims);
    
	h_t = params(:,1,t+start-1)+params(:,2,t+start-1).*(rate_return(t+start-1)-params(:,4,t+start-1)).^2 + params(:,3,t+start-1).*h(:,end);
    h_predict(:,t)=h_t;
    
    CI_low_h_t(:,t)=quantile_weighted( h_t,0.025,W(:,t+start));
    CI_up_h_t(:,t)=quantile_weighted( h_t,0.975,W(:,t+start));
    Predict_median_h_t(:,t)=quantile_weighted(h_t,0.5,W(:,t+start));
    
    [likelihood_predict_TureData_var(t),x] = ksdensity(h_t,h_garch(start+t),'Weights',W(:,start+t));
     
end

ELPD_LFO_true_var = sum(log(likelihood_predict_TureData_var));

save('garch_BayesPostPredict_SimulatedData_EPLD_var.mat');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [loglikelihood,h] = garch_loglike_parallel(params,data,mylims)
% Computing the likelihood and conditional variances of the time series under GARCH dynamics, given observed data and model parameters
%====================================================================================================================

[N,d]=size(params);

% Transforming back to original scale;
for j = 1:d
    params(:,j) = (mylims(j,2)'.*exp(params(:,j))+mylims(j,1)')./(exp(params(:,j))+1);
end

%%%%%%%%%%%%%%%%%%%
%SETTING PARAMETERS
%%%%%%%%%%%%%%%%%%%
a0 = params(:,1); %alpha_0
a1 = params(:,2); %alpha_1
b1 = params(:,3); %beta_1
r_bar = params(:,4); %mu

n = length(data);
h = zeros(N,n);

t1=10e-6;

h(:,1) = max(a0./(1-a1-b1),t1);

logl = zeros(N,1);
logl = -0.5*log(h(:,1)) - 0.5*(data(1)-r_bar).^2./h(:,1) - 0.5*log(2*pi*ones(N,1));

for t = 2:n
    h(:,t) = a0+a1.*(data(t-1)-r_bar).^2 + b1.*h(:,t-1);            
    logl_tmp = -0.5*log(h(:,t)) - 0.5*(data(t)-r_bar).^2./h(:,t) - 0.5*log(2*pi*ones(N,1));
    logl = logl + logl_tmp;
end

loglikelihood = logl;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


