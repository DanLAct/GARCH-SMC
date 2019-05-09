function [h_predict, CI_low_h_t, CI_up_h_t, Predict_median_h_t] = bege_BayesPostPredict_volatility(N, theta, theta_particle, W)
%One-step ahead forecast of conditional variances under BEGE model for Real Data, with using the output of parameters sampled from Data Annealing SMC
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

%loading monthly S&P stock log return
load('Data.mat');
rate_return = MonthlyReturns2018;
rate_return(isnan(rate_return)) = 0;

%Ranges for parameters transformation
mylims = zeros(11,2);
mylims(:,1) = 1e-4.*ones(11,1);
mylims(9,1) = -0.2;
mylims(11,1) = -0.9;
mylims(:,2) = [0.5;0.3;0.99;0.5;0.5;1;0.3;0.99;0.1;0.75;0.9];

T = length(rate_return); % total number of data
start=200; % Starting time point for making prediction; the time point for the 1st predicted conditional variance is: t = start+1
PT=T-start;

CI_low_h_t=zeros(1,PT);
CI_up_h_t=zeros(1,PT);
Predict_median_h_t=zeros(1,PT);
t1=10e-1;
params=theta;
h_predict = zeros(N,PT);

parfor t=1:PT
    fprintf('Just starting with the %ith prediction.\n',t);
    
    [~,ptseries,ntseries]=bege_gjrgarch_likelihood_parallel(theta_particle(:,:,t+start-1),rate_return(1:t+start-1),mylims);
    
    p_bar = params(:,1,t+start-1);
    tp = params(:,2,t+start-1);
    rho_p = params(:,3,t+start-1);
    phi_pp = params(:,4,t+start-1);
    phi_pn = params(:,5,t+start-1);
    n_bar = params(:,6,t+start-1);
    tn = params(:,7,t+start-1);
    rho_n = params(:,8,t+start-1);
    phi_np = params(:,9,t+start-1);
    phi_nn = params(:,10,t+start-1);
    r_bar = params(:,11,t+start-1);
    
    p_t=ptseries(:,end);
    n_t=ntseries(:,end);
    
    %Forecast BEGE - using information at time 't-1' to predict time 't'
    inds_neg=find(rate_return(t+start-1)-r_bar<0);
    inds_pos=find(rate_return(t+start-1)-r_bar>=0);
    
    if isempty(inds_neg)==0
        p_t(inds_neg)=max(p_bar(inds_neg)+rho_p(inds_neg).*ptseries(inds_neg,end)+...
            phi_pn(inds_neg).*(((rate_return(t+start-1)-r_bar(inds_neg)).^2)./(2*(tp(inds_neg).^2))),t1);
        n_t(inds_neg)=max(n_bar(inds_neg)+rho_n(inds_neg).*ntseries(inds_neg,end)+...
            phi_nn(inds_neg).*(((rate_return(t+start-1)-r_bar(inds_neg)).^2)./(2*(tn(inds_neg).^2))),t1);
    end
    
    if isempty(inds_pos)==0
        p_t(inds_pos)=max(p_bar(inds_pos)+rho_p(inds_pos).*ptseries(inds_pos,end)+...
            phi_pp(inds_pos).*(((rate_return(t+start-1)-r_bar(inds_pos)).^2)./(2*(tp(inds_pos).^2))),t1);
        n_t(inds_pos)=max(n_bar(inds_pos)+rho_n(inds_pos).*ntseries(inds_pos,end)+...
            phi_np(inds_pos).*(((rate_return(t+start-1)-r_bar(inds_pos)).^2)./(2*(tn(inds_pos).^2))),t1);
    end
    
    h_t = tp.^2.*p_t+tn.^2.*n_t;
    h_predict(:,t) = h_t;
    
    CI_low_h_t(:,t)=quantile_weighted( h_predict(:,t),0.025,W(:,t+start));
    CI_up_h_t(:,t)=quantile_weighted( h_predict(:,t),0.975,W(:,t+start));
    Predict_median_h_t(:,t)=quantile_weighted(h_predict(:,t),0.5,W(:,t+start));

end

save('results_BayesPostPredict_RealData_var.mat');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [loglikelihood,ptseries,ntseries] = bege_gjrgarch_likelihood_parallel(params,data,mylims)
% Computing the likelihood and conditional variances of the time series under BEGE-GJR-GARCH dynamics, given observed data and model parameters
%====================================================================================================================

[N,d]=size(params);

% Transforming back to original scale;
for j = 1:d
    params(:,j) = (mylims(j,2)'.*exp(params(:,j))+mylims(j,1)')./(exp(params(:,j))+1);
end

%%%%%%%%%%%%%%%%%%%
%SETTING PARAMETERS
%%%%%%%%%%%%%%%%%%%
r_bar = params(:,11); % (mu)
p_bar=params(:,1); % (p_0)
tp=params(:,2); % (sigma_p)
rho_p=params(:,3); %
phi_pp=params(:,4); %
phi_pn=params(:,5); %
n_bar=params(:,6); % (n_0)
tn=params(:,7); % (sigma_n)
rho_n=params(:,8); %
phi_np=params(:,9); %
phi_nn=params(:,10); %


%Computing underlying pt and nt processes
ptseries=zeros(N,length(data));
ntseries=zeros(N,length(data));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%COMPUTING THE LOG-LIKELIHOOD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

loglikelihood=zeros(N,1);
t1=10e-1;

previous_p=max(p_bar./(1-rho_p-(phi_pp+phi_pn)/2),t1);
previous_n=max(n_bar./(1-rho_n-(phi_np+phi_nn)/2),t1);
ptseries(:,1)=max(previous_p,t1);
ntseries(:,1)=max(previous_n,t1);

loglikelihood=loglikelihood+loglikedgam_parallel(data(1)-r_bar,ptseries(:,1),ntseries(:,1),tp,tn,0.001);

p_t=zeros(N,1);
n_t=zeros(N,1);
for t=2:length(data)
    inds_neg=find(data(t-1)-r_bar<0);
    inds_pos=find(data(t-1)-r_bar>=0);
    
    if isempty(inds_neg)==0
       p_t(inds_neg)=max(p_bar(inds_neg)+rho_p(inds_neg).*previous_p(inds_neg)+...
           phi_pn(inds_neg).*(((data(t-1)-r_bar(inds_neg)).^2)./(2*(tp(inds_neg).^2))),t1);
       n_t(inds_neg)=max(n_bar(inds_neg)+rho_n(inds_neg).*previous_n(inds_neg)+...
           phi_nn(inds_neg).*(((data(t-1)-r_bar(inds_neg)).^2)./(2*(tn(inds_neg).^2))),t1);
    end
    if isempty(inds_pos)==0
       p_t(inds_pos)=max(p_bar(inds_pos)+rho_p(inds_pos).*previous_p(inds_pos)+...
           phi_pp(inds_pos).*(((data(t-1)-r_bar(inds_pos)).^2)./(2*(tp(inds_pos).^2))),t1);
       n_t(inds_pos)=max(n_bar(inds_pos)+rho_n(inds_pos).*previous_n(inds_pos)+...
           phi_np(inds_pos).*(((data(t-1)-r_bar(inds_pos)).^2)./(2*(tn(inds_pos).^2))),t1);
    end
    
    obs=data(t)-r_bar;
    tmp = loglikedgam_parallel(obs,p_t,n_t,tp,tn,0.001);
    loglikelihood=loglikelihood+tmp;
    ptseries(:,t)=p_t;
    ntseries(:,t)=n_t;
    previous_p=p_t;
    previous_n=n_t;
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
