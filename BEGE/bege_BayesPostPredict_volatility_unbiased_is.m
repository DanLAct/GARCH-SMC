function [h_predict, CI_low_h_t, CI_up_h_t, Predict_median_h_t] = bege_BayesPostPredict_volatility_unbiased_is(N, theta, theta_particle, W)
%One-step ahead forecast of conditional variances under BEGE model for Real Data, with using the output of parameters sampled from Data Annealing SMC
%The likelihood is unbiasedly estimated using importance sampling method
%
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
%ptseries = zeros(N,1);
%ntseries = zeros(N,1);

for t=1:PT
    fprintf('Just starting with the %ith prediction.\n',t);
    ptseries_prop = zeros(N,length(rate_return(1:t+start-1)));
    ntseries_prop = zeros(N,length(rate_return(1:t+start-1)));
    parfor i=1:N
        [~,ptseries_prop(i,:),ntseries_prop(i,:)] = bege_gjrgarch_likelihood_unbiased_is(theta_particle(i,:,t+start-1),rate_return(1:t+start-1),mylims);
        
    end
    
    ptseries = ptseries_prop;
    ntseries = ntseries_prop;
    
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
%save('results_BayesPostPredict_RealData_unbiased_is_var.mat');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [loglikelihood,ptseries,ntseries] = bege_gjrgarch_likelihood_unbiased_is(params,data,mylims)
% Computing the likelihood of the time series under BEGE-GJR-GARCH dynamics, given observed data and model parameters
%====================================================================================================================

% Transforming back to original scale;
for j = 1:length(params)
    params(j) = (mylims(j,2)'.*exp(params(j))+mylims(j,1)')/(exp(params(j))+1);
end

%%%%%%%%%%%%%%%%%%%
%SETTING PARAMETERS
%%%%%%%%%%%%%%%%%%%
r_bar = params(11); % (mu)
p_bar=params(1); % (p_0)
tp=params(2); % (sigma_p)
rho_p=params(3); %
phi_pp=params(4); %
phi_pn=params(5); %
n_bar=params(6); % (n_0)
tn=params(7); % (sigma_n)
rho_n=params(8); %
phi_np=params(9); %
phi_nn=params(10); %


%Computing underlying pt and nt processes
ptseries=zeros(length(data),1);
ntseries=zeros(length(data),1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%COMPUTING THE LOG-LIKELIHOOD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

loglikelihood=0;
t1=10e-1;

previous_p=max(p_bar/(1-rho_p-(phi_pp+phi_pn)/2),t1);
previous_n=max(n_bar/(1-rho_n-(phi_np+phi_nn)/2),t1);
ptseries(1)=max(previous_p,t1);
ntseries(1)=max(previous_n,t1);

loglikelihood=loglikelihood+loglikedgam_unbiased_is(data(1)-r_bar,ptseries(1),ntseries(1),tp,tn);

for t=2:length(data)
    if ((data(t-1)-r_bar)<0)
        p_t=max(p_bar+rho_p*previous_p+...
            phi_pn*(((data(t-1)-r_bar)^2)/(2*(tp^2))),t1);
        n_t=max(n_bar+rho_n*previous_n+...
            phi_nn*(((data(t-1)-r_bar)^2)/(2*(tn^2))),t1);
    else
        p_t=max(p_bar+rho_p*previous_p+...
            phi_pp*(((data(t-1)-r_bar)^2)/(2*(tp^2))),t1);
        n_t=max(n_bar+rho_n*previous_n+...
            phi_np*(((data(t-1)-r_bar)^2)/(2*(tn^2))),t1);
    end
    obs=data(t)-r_bar;
    tmp = loglikedgam_unbiased_is(obs,p_t,n_t,tp,tn);
    loglikelihood=loglikelihood+tmp;
    
    ptseries(t)=p_t;
    ntseries(t)=n_t;
    previous_p=p_t;
    previous_n=n_t;
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function loglikedgam = loglikedgam_unbiased_is(z,p,n,tp,tn)
% This function unbiasedly estimates the likelihood of an observation under
% the BEGE density by using importance sampling.
% The likelihood can be exactly computed in the cases of p=n=1; p=1&n>1;
% n=1&p>1.
%
% Input:
% z - the point at which the pdf is evaluated
% p - good environment shape parameter
% n - bad environment shape parameter
% tp - good environment scale parameter
% tn - bad environment scale parameter
%
% Output:
% loglikedgam - the loglikelihood of the observations


wp_bar = -p*tp;
wn_bar = -n*tn;
sigma = 1/tp + 1/tn;
delta = max(wp_bar, wn_bar + z);
N = 1000;

if p==1 && n==1
    loglikedgam = -log(tp) - log(tn) + wp_bar/tp + (z+wn_bar)/tn - log(sigma)...
        - sigma*delta;
    return;
end

if p==1 && n>1
    
    if (delta == wn_bar + z)
        loglikedgam = -log(tp)  - n*log(tn) + wp_bar/tp - (z + wn_bar)/tp  - n*log(sigma);
        return;
    else
        loglikedgam = -log(tp)  - n*log(tn) + wp_bar/tp - (z + wn_bar)/tp  - n*log(sigma) + log(1 - gamcdf(delta - (wn_bar + z), n, 1/sigma));
        return;
    end
    
end

if n==1 && p>1
    if (delta == wp_bar)
        loglikedgam =  -log(tn) - p*log(tp)  + (z + wn_bar)/tn - wp_bar/tn  - p*log(sigma);
        return;
    else
        loglikedgam =  -log(tn) - p*log(tp)  + (z + wn_bar)/tn - wp_bar/tn  - p*log(sigma) + log(1 - gamcdf(delta - wp_bar, p, 1/sigma));
        return;
    end
end

% n>1 && p>1
bneg_mode = sigma*(z+wn_bar+wp_bar)+(p+n-2);
constant = sigma*(z*wp_bar + wp_bar*wn_bar) + (p-1)*(z+wn_bar) + (n-1)*wp_bar;
mode = (bneg_mode + sqrt(bneg_mode^2 - 4*sigma*constant))/(2*sigma);

variance = -1/((1-p)/(mode-wp_bar)^2 + (1-n)/(mode-z-wn_bar)^2);
bneg = 2*variance + (mode - delta)^2;
a = (bneg + sqrt(bneg^2 - 4*variance^2))/(2*variance);
b = (mode - delta)/(a-1);
wp = gamrnd(a,b,N,1) + delta;
log_target = -gammaln(p) - p*log(tp) + (p-1)*log(wp-wp_bar) - (wp-wp_bar)/tp...
    - gammaln(n) - n*log(tn) + (n-1)*log(wp-z-wn_bar) - (wp-z-wn_bar)/tn;
log_importance = -a*log(b) - gammaln(a) +(a-1)*log(wp - delta) -(wp - delta)/b;
logw = log_target - log_importance;
loglikedgam = -log(N) + logsumexp(logw);

end