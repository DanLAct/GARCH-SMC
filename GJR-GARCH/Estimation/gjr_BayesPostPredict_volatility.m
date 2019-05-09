function [h_predict, CI_low_h_t, CI_up_h_t, Predict_median_h_t] = gjr_BayesPostPredict_volatility(N, theta, theta_particle, W)
%One-step ahead forecast of conditional variances under GJR-GARCH(1,1) model for Real Data, with using the output of parameters sampled from Data Annealing SMC
%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%
% N                  - Size of population of particles
% theta              - Parameters sampled from Data Annealing SMC: N samples from different amount of available observations: y_{1:t}, t=1,..,T
% theta_particle     - Transformed theta
% W                  - The weights of weighted samples, corresponding to the above thetas/theta_particles

%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%
% h_predict            - N samples of the one-step ahead forecast of conditional variances
% CI_low_h_t           - 0.025 quantile of the above posterior predictive sample of the conditional variances
% CI_up_h_t            - 0.975 quantile of the above posterior predictive sample of the conditional variances
% Predict_median_h_t   - Median of the above posterior predictive sample of the conditional variances
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%load('Data for Prediction_gjr_unif.mat');

%loading monthly S&P stock log return
load('Data.mat');
rate_return = MonthlyReturns2018;
rate_return(isnan(rate_return)) = 0;

%loading monthly S&P stock log return
mylims = zeros(5,2);
mylims(:,1) = zeros(5,1);
mylims(4,1) = 0.5;
mylims(5,1) = -0.9;
mylims(:,2) = [0.3; 0.3; 0.3; 0.99; 0.9];

T = length(rate_return); % total number of data
start=200; % Starting time point for making prediction; the time point for the 1st predicted conditional variance is: t = start+1
PT=T-start;

CI_low_h_t=zeros(1,PT);
CI_up_h_t=zeros(1,PT);
Predict_median_h_t=zeros(1,PT);
params=theta;
h_predict = zeros(N,PT);

parfor t=1:PT
    fprintf('Just starting with the %ith prediction.\n',t);
    
    [~,h] = gjrloglike_parallel_unif(theta_particle(:,:,t+start-1),rate_return(1:t+start-1),mylims);
    
    h_t = params(:,1,t+start-1)+params(:,2,t+start-1).*(rate_return(t+start-1)-params(:,5,t+start-1)).^2 ...
        + params(:,3,t+start-1).*((rate_return(t+start-1)-params(:,5,t+start-1))<0).*(rate_return(t+start-1)-params(:,5,t+start-1)).^2 ...
        + params(:,4,t+start-1).*h(:,end);
    h_predict(:,t)=h_t;
    
    CI_low_h_t(:,t)=quantile_weighted( h_t,0.025,W(:,t+start));
    CI_up_h_t(:,t)=quantile_weighted( h_t,0.975,W(:,t+start));
    Predict_median_h_t(:,t)=quantile_weighted(h_t,0.5,W(:,t+start));
    
end
save('gjr_BayesPostPredict__RealData_var.mat');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [loglikelihood,h] = gjrloglike_parallel_unif(params,data,mylims)
% Computing the likelihood of the time series under GJR-GARCH dynamics, given observed data and model parameters
%====================================================================================================================

[N,d]=size(params);

% Transforming back to original scale;
for j = 1:d
    params(:,j) = (mylims(j,2)'.*exp(params(:,j))+mylims(j,1)')./(exp(params(:,j))+1);
end

%%%%%%%%%%%%%%%%%%%
%SETTING PARAMETERS
%%%%%%%%%%%%%%%%%%%
h0 = params(:,1); %alpha_0
phi_p = params(:,2); %phi
phi_n = params(:,3); %phi_{-}
rho_h = params(:,4); %beta_sigma
r_bar = params(:,5); %mu

t1=10e-6;

n = length(data);
h = zeros(N,n);

h(:,1) = max(h0./(1-phi_p-phi_n./2-rho_h),t1);

logl=zeros(N,1);
logl = - .5*log(h(:,1)) - .5*(data(1)-r_bar).^2./h(:,1) -.5*log(2*pi*ones(N,1));

for t = 2:n
    inds_neg = find(data(t-1)-r_bar<0);
    inds_pos = find(data(t-1)-r_bar>0);    
    h(inds_pos,t) = h0(inds_pos) + rho_h(inds_pos).*h(inds_pos,t-1)+ phi_p(inds_pos).*(data(t-1)-r_bar(inds_pos)).^2;
    h(inds_neg,t) = h0(inds_neg) + rho_h(inds_neg).*h(inds_neg,t-1)+ phi_p(inds_neg).*(data(t-1)-r_bar(inds_neg)).^2+ phi_n(inds_neg).*(data(t-1)-r_bar(inds_neg)).^2;
    logl_tmp = - .5*log(h(:,t)) - .5*(data(t)-r_bar).^2./h(:,t) -.5*log(2*pi*ones(N,1));
    logl=logl+logl_tmp;
    
end

loglikelihood=logl;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

