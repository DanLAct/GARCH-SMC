function [theta, theta_particle, loglike, logprior, gamma, log_evidence] = SMC_RW_LikeAnneal_ParVec_unbiased_is(N)
%SMC utilised vectorisation and parallelisation for estimating BEGE model's
%parameters;
%The sequence of distributions constructed by using Likelihood Annealing method
%The likelihood is unbiasedly estimated using importance sampling method

%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%
% N                  - Size of population of particles

%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%
% theta              - N samples from each temperature
% theta_particle     - N transformed samples from each temperature
% loglike            - Log likelihood of the BEGE model, corresponding to the above thetas
% logprior           - Log prior of the BEGE model, corresponding to the above thetas
% gamma              - The temperatures from likelihood annealing strategy
% log_evidence       - The estimate of log evidence used for model selection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%loading monthly S&P stock log return
load('Data.mat');
rate_return = MonthlyReturns2018;
rate_return(isnan(rate_return)) = 0;

%Setting ranges for parameters to do transformation
mylims = zeros(11,2);
mylims(:,1) = 1e-4.*ones(11,1);
mylims(9,1) = -0.2;
mylims(11,1) = -0.9;
mylims(:,2) = [0.5;0.3;0.99;0.5;0.5;1;0.3;0.99;0.1;0.75;0.9];

%Starting parallel pool
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
quthpc = parcluster('local');
parpool(quthpc);
poolsize = quthpc.NumWorkers;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initialising
t = 1;
Num_Parameter = 11; % dimension of theta
log_evidence = 0;
gamma = 0; % temperatures
logw_previous=log(ones(N,1)*(1/N));
h = 0.1:0.1:1; % For choosing optimal scales and thus to reduce the number of MCMC repeats.

%Transform theta
theta = zeros(N,Num_Parameter);
theta_particle = zeros(N,Num_Parameter);
for j = 1:Num_Parameter
    theta(:,j) = unifrnd(mylims(j,1),mylims(j,2),[N,1]);
    theta_particle(:,j) = log((theta(:,j) - mylims(j,1))./(mylims(j,2) - theta(:,j)));
end

%calculating initial log prior
logprior = logPrior_parallel_fv(theta_particle,mylims);


inds = 1:N;
while isempty(inds)==0 %this while loop is used to eliminated any infinity value from simulation
    for j=1:Num_Parameter
        theta(inds,j) = unifrnd(mylims(j,1),mylims(j,2),[length(inds),1]);
        theta_particle(inds,j) = log((theta(inds,j) - mylims(j,1))./(mylims(j,2) - theta(inds,j)));
    end
    logprior(inds) = logPrior_parallel_fv(theta_particle(inds,:),mylims);
    inds = find(isinf(logprior));
end

%calculating initial log likelihood
[loglike, ~, ~] = bege_gjrgarch_likelihood_parallel_unbiased_is(theta_particle,rate_return,mylims);

while gamma(t)< 1
    %Testing gammavar=1
    logw = logw_previous+(1-gamma(t))*loglike(:,t);
    w = logw-max(logw); %stability
    w = exp(w);
    w = w/sum(w);
    ESS1 = 1/sum(w.^2);
    
    %Choosing next temperature
    if (ESS1 >= N/2)
        gamma(t+1) = 1;
    else
        %use bisection method to find the sequence of gamma
        fun=@(gamma_Current)Gamma(gamma_Current,gamma(t),N,loglike(:,t),logw_previous);
        interval=[gamma(t) 1];
        gamma_Current=fzero(fun,interval);
        gamma(t+1)=gamma_Current;
    end
    fprintf('The current temperature is %.3f.\n',gamma(t+1));
    
    %Subsitute the value of just calculated gamma(t)
    logw = logw_previous + (gamma(t+1)- gamma(t))*loglike(:,t);
    log_evidence=log_evidence + logsumexp(logw);
    w = logw-max(logw);
    w = exp(w);
    w = w/sum(w);
    
    % (systematic resampling)
    u=rand;
    indices = zeros(N,1);
    cumsum_w = cumsum(w);
    for i = 1:N
        A = find(u < cumsum_w);
        indices(i) = A(1);
        u=u+1/N;
        if u>1
            u=u-1;
        end
    end
    theta_particle(:,:,t) = theta_particle(indices,:,t);
    loglike(:,t) = loglike(indices,t);
    logprior(:,t) = logprior(indices,t);
    
    %(Move with MCMC Kernel)
    cov_rw = cov(theta_particle(:,:,t)); %covariance of resampled
    Cov_inv=cov_rw^(-1); % inverse of the above covariance
    
    %compute mahalanobis distance from the Sample before moving
    [~,dists]=rangesearch(theta_particle(:,:,t),theta_particle(:,:,t),inf,'distance','mahalanobis','cov',cov_rw);
    dists=cell2mat(dists);
    median_dist=median(dists(:));
    
    h_ind=mod(randperm(N),length(h))'+1;
    h_all=h(h_ind);
    ESJD=zeros(N,1);
    
    %MVN RW
    parfor i=1:N
        theta_particle_prop(i,:) = mvnrnd(theta_particle(i,:,t),h_all(i)^2*cov_rw);
    end
    logprior_prop = logPrior_parallel_fv(theta_particle_prop,mylims);
    
    inds = find(isinf(logprior_prop)==0);
    len_inds = length(inds);
    
    percore = ceil(len_inds/poolsize);
    parfor core = 1:poolsize
        current_core = percore*(core-1)+1: min(percore*core,len_inds);
        [loglike_prop_cell{core},~,~] = bege_gjrgarch_likelihood_parallel_unbiased_is(theta_particle_prop(inds(current_core),:),rate_return,mylims);
    end
    loglike_prop = loglike_prop_cell{1};
    for core = 2:poolsize
        loglike_prop = [loglike_prop; loglike_prop_cell{core}];
    end
    
    log_mh = gamma(t+1)*loglike_prop - gamma(t+1)*loglike(inds,t) + logprior_prop(inds) - logprior(inds,t);
    
    acc_probs = -inf*ones(N,1);
    acc_probs(inds) = exp(log_mh);
    
    for i=1:length(inds)
        ESJD(inds(i))=((theta_particle(inds(i),:,t)-theta_particle_prop(inds(i),:))*Cov_inv*(theta_particle(inds(i),:,t)-theta_particle_prop(inds(i),:))').^(1/2)*acc_probs(inds(i));
    end
    
    toacc_sub = find(rand(len_inds,1)<acc_probs(inds));
    toacc = inds(toacc_sub);
    theta_particle(toacc,:,t) = theta_particle_prop(toacc,:);
    loglike(toacc,t) = loglike_prop(toacc_sub);
    logprior(toacc,t) = logprior_prop(toacc);
    
    median_ESJD = accumarray(h_ind,ESJD,[],@median);%Median value of ESJD for different h indices from 1 to 10
    ind=median_ESJD==max(median_ESJD);
    h_opt = h(ind);
    fprintf('the scale is %f\n',h_opt);
    
    dist_move = zeros(N,1);
    belowThreshold = true;
    R_move = 0;
    
    
    % Getting number of remaining MCMC repeats
    % Performing remaining repeats
    while belowThreshold
        R_move=R_move+1;
        
        theta_particle_prop= mvnrnd(theta_particle(:,:,t),h_opt^2*cov_rw);
        
        logprior_prop = logPrior_parallel_fv(theta_particle_prop,mylims);
        
        inds = find(isinf(logprior_prop)==0);
        len_inds = length(inds);
        
        percore = ceil(len_inds/poolsize);
        parfor core = 1:poolsize
            current_core = percore*(core-1)+1: min(percore*core,len_inds);
            [loglike_prop_cell{core},~,~] = bege_gjrgarch_likelihood_parallel_unbiased_is(theta_particle_prop(inds(current_core),:),rate_return,mylims);
        end
        loglike_prop = loglike_prop_cell{1};
        for core = 2:poolsize
            loglike_prop = [loglike_prop; loglike_prop_cell{core}];
        end
        
        log_mh = gamma(t+1)*loglike_prop - gamma(t+1)*loglike(inds,t) + logprior_prop(inds) - logprior(inds,t);
        
        acc_probs = exp(log_mh);
        
        toacc_sub = find(rand(len_inds,1)<acc_probs);
        toacc = inds(toacc_sub);
        
        for i=1:length(toacc)
            dist_move(toacc(i)) = dist_move(toacc(i)) + sqrt((theta_particle(toacc(i),:,t)-theta_particle_prop(toacc(i),:))*Cov_inv*(theta_particle(toacc(i),:,t)-theta_particle_prop(toacc(i),:))');
        end
        
        theta_particle(toacc,:,t) = theta_particle_prop(toacc,:);
        loglike(toacc,t) = loglike_prop(toacc_sub);
        logprior(toacc,t) = logprior_prop(toacc);
        
        if sum(dist_move>median_dist)>=ceil(0.5*N)
            belowThreshold = false;
        end
    end
    fprintf('the value of R_move was %d\n',R_move);
    
    theta_particle(:,:,t+1) = theta_particle(:,:,t);
    loglike(:,t+1) = loglike(:,t);
    logprior(:,t+1) = logprior(:,t);
    
    t = t+1;
end

%Transforming back to original scale
theta = zeros(N,Num_Parameter,size(theta_particle,3));
for j=1:Num_Parameter
    theta(:,j,:) = (mylims(j,2)'.*exp(theta_particle(:,j,:))+mylims(j,1)')./(exp(theta_particle(:,j,:))+1);
end

delete(gcp);%shut down parallel pool
%save('results_bege_LikeAnneal_is.mat');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [logprior] = logPrior_parallel_fv(theta_particle,mylims)
% Computes the log prior (takes transformed parameters as input)

sumA =  [-eye(11); eye(11); 0 0 1 0.5 0.5 0 0 0 0 0 0; 0 0 0 0 0 0 0 1 0.5 0.5 0];

sumB =  [zeros(10,1); 0.9;...
    0.5;0.3;0.99;0.5;0.5;1;0.3;0.99;0.1;0.75;0.9; 0.995;0.995];
sumB(9) = 0.2;

[N,d]=size(theta_particle);

% Transforming back to original scale
theta = theta_particle;
for j = 1:d
    theta(:,j) = (mylims(j,2)'.*exp(theta(:,j))+mylims(j,1)')./(exp(theta(:,j))+1);
end

logprior = -inf*ones(N,1);
inds=find(all(sumA*theta'<=sumB*ones(1,N))==1);
logprior(inds)=sum(-theta_particle(inds,:)-2*log(1+exp(-theta_particle(inds,:))),2);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [loglikelihood,ptseries,ntseries] = bege_gjrgarch_likelihood_parallel_unbiased_is(params,data,mylims)
% Computing unbiased estimator of the likelihood of the time series under BEGE-GJR-GARCH dynamics, given observed data and model parameters
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

loglikelihood=loglikelihood+loglikedgam_parallel_unbiased_is(data(1)-r_bar,ptseries(:,1),ntseries(:,1),tp,tn);

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
    tmp = loglikedgam_parallel_unbiased_is(obs,p_t,n_t,tp,tn);
    loglikelihood=loglikelihood+tmp;
    ptseries(:,t)=p_t;
    ntseries(:,t)=n_t;
    previous_p=p_t;
    previous_n=n_t;
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function loglikedgam = loglikedgam_parallel_unbiased_is(z,p,n,tp,tn)
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


wp_bar = -p.*tp;
wn_bar = -n.*tn;
sigma = 1./tp + 1./tn;
delta = max(wp_bar, wn_bar + z);
N = 1000;
len = length(p);
loglikedgam = zeros(len,1);

np = [n p];
inds_n = np(:,1)==1;
inds_p = np(:,2)==1;
inds_one = inds_n == 1 & inds_p == 1;
inds_p_one = inds_n == 0 & inds_p == 1;
inds_p_one_1 = inds_n == 0 & inds_p == 1 & delta == wn_bar + z;
inds_n_one = inds_n == 1 & inds_p == 0;
inds_n_one_1 = inds_n == 1 & inds_p == 0 & delta == wp_bar;
inds_p_n = inds_n == 0 & inds_p == 0;

if isempty(inds_one)==0%p==1 && n==1
    loglikedgam(inds_one) = -log(tp(inds_one)) - log(tn(inds_one)) + wp_bar(inds_one)./tp(inds_one)...
        + (z(inds_one)+wn_bar(inds_one))./tn(inds_one) - log(sigma(inds_one))...
        - sigma(inds_one).*delta(inds_one);
end

if isempty(inds_p_one)==0 %p==1 && n>1
    
    loglikedgam(inds_p_one_1) = -log(tp(inds_p_one_1))  - n(inds_p_one_1).*log(tn(inds_p_one_1))...
        + wp_bar(inds_p_one_1)./tp(inds_p_one_1) - (z(inds_p_one_1) + wn_bar(inds_p_one_1))./tp(inds_p_one_1)...
        - n(inds_p_one_1).*log(sigma(inds_p_one_1));
    loglikedgam(~inds_p_one_1) = -log(tp(~inds_p_one_1))  - n(~inds_p_one_1).*log(tn(~inds_p_one_1))...
        + wp_bar(~inds_p_one_1)./tp(~inds_p_one_1) - (z(~inds_p_one_1) + wn_bar(~inds_p_one_1))./tp(~inds_p_one_1)...
        - n(~inds_p_one_1).*log(sigma(~inds_p_one_1)) + log(1 - gamcdf(delta(~inds_p_one_1) - (wn_bar(~inds_p_one_1) + z(~inds_p_one_1)), n(~inds_p_one_1), 1./sigma(~inds_p_one_1)));
end

if isempty(inds_n_one)==0 %n==1 && p>1
    
    loglikedgam(inds_n_one_1) =  -log(tn(inds_n_one_1)) - p(inds_n_one_1).*log(tp(inds_n_one_1))...
        + (z(inds_n_one_1) + wn_bar(inds_n_one_1))./tn(inds_n_one_1) - wp_bar(inds_n_one_1)./tn(inds_n_one_1)...
        - p(inds_n_one_1).*log(sigma(inds_n_one_1));
    
    loglikedgam(~inds_n_one_1) =  -log(tn(~inds_n_one_1)) - p(~inds_n_one_1).*log(tp(~inds_n_one_1))...
        + (z(~inds_n_one_1) + wn_bar(~inds_n_one_1))./tn(~inds_n_one_1) - wp_bar(~inds_n_one_1)./tn(~inds_n_one_1)...
        - p(~inds_n_one_1).*log(sigma(~inds_n_one_1)) + log(1 - gamcdf(delta(~inds_n_one_1) - wp_bar(~inds_n_one_1), p(~inds_n_one_1), 1./sigma(~inds_n_one_1)));
end

if isempty(inds_p_n)==0 %n>1 && p>1
    bneg_mode = sigma(inds_p_n).*(z(inds_p_n)+wn_bar(inds_p_n)+wp_bar(inds_p_n))+(p(inds_p_n)+n(inds_p_n)-2);
    constant = sigma(inds_p_n).*(z(inds_p_n).*wp_bar(inds_p_n) + wp_bar(inds_p_n).*wn_bar(inds_p_n))...
        + (p(inds_p_n)-1).*(z(inds_p_n)+wn_bar(inds_p_n)) + (n(inds_p_n)-1).*wp_bar(inds_p_n);
    mode = (bneg_mode + sqrt(bneg_mode.^2 - 4*sigma(inds_p_n).*constant))./(2*sigma(inds_p_n));
    
    variance = -1./((1-p(inds_p_n))./(mode-wp_bar(inds_p_n)).^2 + (1-n(inds_p_n))./(mode-z(inds_p_n)-wn_bar(inds_p_n)).^2);
    bneg = 2*variance + (mode - delta(inds_p_n)).^2;
    a = (bneg + sqrt(bneg.^2 - 4*variance.^2))./(2*variance);
    b = (mode - delta(inds_p_n))./(a-1);
    length_inds_p_n = length(find(inds_p_n == 1));
    wp = zeros(length_inds_p_n,N);
    for i=1:length_inds_p_n
        wp(i,:) = gamrnd(a(i),b(i),1,N);
    end
    wp = wp + delta(inds_p_n);
    log_target = -gammaln(p(inds_p_n)) - p(inds_p_n).*log(tp(inds_p_n)) + (p(inds_p_n)-1).*log(wp-wp_bar(inds_p_n))...
        - (wp-wp_bar(inds_p_n))./tp(inds_p_n) - gammaln(n(inds_p_n)) - n(inds_p_n).*log(tn(inds_p_n))...
        + (n(inds_p_n)-1).*log(wp-z(inds_p_n)-wn_bar(inds_p_n)) - (wp-z(inds_p_n)-wn_bar(inds_p_n))./tn(inds_p_n);
    log_importance = -a.*log(b) - gammaln(a) +(a-1).*log(wp - delta(inds_p_n)) -(wp - delta(inds_p_n))./b;
    logw = log_target - log_importance;
    loglikedgam(inds_p_n) = -log(N) + logsumexp(logw')';
end
end



