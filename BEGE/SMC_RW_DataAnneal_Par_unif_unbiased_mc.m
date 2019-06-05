function [theta, theta_particles, loglike, logprior, W, log_evidence] = SMC_RW_DataAnneal_Par_unif_unbiased_mc(N)
%SMC utilised parallelisation for estimating BEGE model's parameters;
%The sequence of distributions constructed by using Data Annealing method
%The likelihood is unbiasedly estimated using Monte Carlo Integration
%
%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%
% N                  - Size of population of particles

%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%
% theta              - N samples from currently available observations: y_{1:t}, t=1,..,T
% theta_particle     - N transformed samples from currently available observations
% loglike            - Log likelihood of the BEGE model, corresponding to the above thetas
% logprior           - Log prior of the BEGE model, corresponding to the above thetas
% W                  - The weights of weighted samples, corresponding to the above thetas/theta_particles
% log_evidence       - The estimate of log evidence used for model selection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%loading monthly S&P stock log return
load('Data.mat');
rate_return = MonthlyReturns2018;
rate_return(isnan(rate_return)) = 0;

%Setting ranges for parameters to do transformation
mylims = zeros(11,2);
mylims(:,1) = zeros(11,1);
mylims(9,1) = -0.2;
mylims(11,1) = -0.9;
mylims(:,2) = [0.5;0.3;0.99;0.5;0.5;1;0.3;0.99;0.1;0.75;0.9];

%Initialising
Num_Parameter=11; % dimension of theta
h = 0.1:0.1:1; % For choosing optimal scales and thus to reduce the number of MCMC repeats.
propotion = 0.5; %to determine the target ESS
T = length(rate_return); % total number of data
theta=zeros(N,Num_Parameter);
theta_particles=zeros(N,Num_Parameter,T);
logprior=zeros(N,T);

%Transform theta
parfor j = 1:Num_Parameter
    theta(:,j,1) = unifrnd(mylims(j,1),mylims(j,2),[N,1]);
    theta_particles(:,j,1) = log((theta(:,j,1) - mylims(j,1))./(mylims(j,2) - theta(:,j,1)));
end

%calculating initial log prior
parfor i = 1:N
    logprior(i,1) = log_prior(theta_particles(i,:,1),mylims);
end

logW = zeros(N,T+1);
logW=-log(N)*ones(N,1);
loglike = zeros(N,T);
loglike_inc = zeros(N,1); %incremental weights; new observation's loglikelihood
ESS = zeros(T,1);
W = (1/N)*ones(N,1); % Initialising weights
logsum_w = zeros(T,1);

inf_ind = 1;
while inf_ind %this while loop is used to eliminated any infinity value from simulation
    inds = find(isinf(logprior(:,1))==1);
    for j = 1:Num_Parameter
        theta(inds,j,1) = unifrnd(mylims(j,1),mylims(j,2),[length(inds),1]);
        theta_particles(inds,j,1) = log((theta(inds,j,1) - mylims(j,1))./(mylims(j,2) - theta(inds,j,1)));
    end
    for i = inds'
        logprior(i,1) = log_prior(theta_particles(i,:,1),mylims);
    end
    inf_ind = any(isinf(logprior(:,1)));
end

% Start sampling
for t=1:T
    fprintf('Just starting with the %ith observation.\n',t);
    rate_return_sub = rate_return(1:t);
    if t==1
        parfor i=1:N
            [~, loglike(i,t),ptseries{i},ntseries{i},~] = bege_gjrgarch_likelihood_unbiased_mc(theta_particles(i,:,t),rate_return(1),mylims);
        end
        logW(:,t+1) = log(W(:,t)) + loglike(:,t);
    else
        parfor i=1:N
            [loglike_inc(i),ptseries{i},ntseries{i}] = bege_gjrgarch_likelihood_danneal_in_unbiased_mc(theta_particles(i,:,t),rate_return(t),mylims,ptseries{i},ntseries{i},rate_return(t-1));
        end
        logW(:,t+1) = log(W(:,t)) + loglike_inc;
        loglike(:,t) = loglike(:,t-1) + loglike_inc;
    end
    
    logsum_w(t) = logsumexp(logW(:,t+1));
    logW(:,t+1)=logW(:,t+1)-max(logW(:,t+1));
    W(:,t+1)=exp(logW(:,t+1))./sum(exp(logW(:,t+1)));
    ESS(t)=1/sum(W(:,t+1).^2);
    
    
    if ESS(t)<propotion*N %Checking if there's a need to resample
        %systematic resampling
        u=rand;
        indices=zeros(N,1);
        cumsum_w=cumsum(W(:,t+1));
        for i=1:N
            A=find(u<cumsum_w);
            indices(i)=A(1);
            u=u+1/N;
            if u>1
                u=u-1;
            end
        end
        theta_particles(:,:,t)=theta_particles(indices,:,t);
        loglike(:,t) = loglike(indices,t);
        logprior(:,t) = logprior(indices,t);
        ptseries = ptseries(indices);
        ntseries = ntseries(indices);
        
        logW(:,t+1) = -log(N)*ones(N,1);
        W(:,t+1)=exp(logW(:,t+1))/sum(exp(logW(:,t+1)));
        
        %(Move with MCMC Kernel)
        cov_rw=cov(theta_particles(:,:,t)); % covariance of resampled
        Cov_inv=cov_rw^(-1); % inverse of the above covariance
        
        %compute mahalanobis distance from the Sample before moving
        [~,dists]=rangesearch(theta_particles(:,:,t),theta_particles(:,:,t),inf,'distance','mahalanobis','cov',cov_rw);
        dists=cell2mat(dists);
        median_dist=median(dists(:));
        
        h_ind=mod(randperm(N),length(h))'+1;
        h_all=h(h_ind);
        ESJD=zeros(N,1);
        acc_prob=zeros(N,1);
        
        %MVN RW
        parfor i=1:N
            
            theta_particles_prop = mvnrnd(theta_particles(i,:,t),h_all(i)^2*cov_rw);
            logprior_prop=log_prior(theta_particles_prop,mylims);
            [~, loglike_prop,ptseries_prop,ntseries_prop,~] = bege_gjrgarch_likelihood_unbiased_mc(theta_particles_prop, rate_return_sub,mylims);
            if isinf(logprior_prop)
                continue;
            end
            
            Alpha=exp(loglike_prop - loglike(i,t) + logprior_prop - logprior(i,t));
            acc_prob(i)=min(1,Alpha);
            
            ESJD(i)=((theta_particles(i,:,t)-theta_particles_prop)*Cov_inv*(theta_particles(i,:,t)-theta_particles_prop)').^(1/2)*acc_prob(i);
            
            if rand<acc_prob(i)
                theta_particles(i,:,t)=theta_particles_prop;
                logprior(i,t)=logprior_prop;
                loglike(i,t)=loglike_prop;
                ptseries{i} = ptseries_prop;
                ntseries{i} = ntseries_prop;
            end
        end
        
        median_ESJD = accumarray(h_ind,ESJD,[],@median);%Median value of ESJD for different h indices from 1 to 10
        ind=median_ESJD==max(median_ESJD);
        h_opt = h(ind);
        fprintf('the scale is %f\n',h_opt);
        
        dist_move = zeros(N,1);
        belowThreshold = true;
        R_move = 0;
        
        %Getting number of remaining MCMC repeats
        %Performing remaining repeats
        while belowThreshold
            R_move=R_move+1;
            
            parfor i=1:N
                theta_particles_prop = mvnrnd(theta_particles(i,:,t),h_opt^2*cov_rw);
                
                logprior_prop=log_prior(theta_particles_prop,mylims);
                [~, loglike_prop,ptseries_prop,ntseries_prop,~] = bege_gjrgarch_likelihood_unbiased_mc(theta_particles_prop, rate_return_sub,mylims);
                if isinf(logprior_prop)
                    continue;
                end
                
                Alpha=exp(loglike_prop - loglike(i,t) + logprior_prop - logprior(i,t));
                acc_prob(i)=min(1,Alpha);
                
                if rand<acc_prob(i)
                    dist_move(i) = dist_move(i) + sqrt((theta_particles(i,:,t)-theta_particles_prop)*Cov_inv*(theta_particles(i,:,t)-theta_particles_prop)');
                    theta_particles(i,:,t)=theta_particles_prop;
                    logprior(i,t)=logprior_prop;
                    loglike(i,t)=loglike_prop;
                    ptseries{i} = ptseries_prop;
                    ntseries{i} = ntseries_prop;
                end
            end
            
            if sum(dist_move>median_dist)>=ceil(0.5*N)
                belowThreshold = false;
            end
        end
        fprintf('the value of R_move was %d\n',R_move);
    end
    if t~=T
        theta_particles(:,:,t+1) = theta_particles(:,:,t);
        loglike(:,t+1) = loglike(:,t);
        logprior(:,t+1) = logprior(:,t);
    end
end

%Transforming back to original scale
theta = theta_particles;
parfor j=1:Num_Parameter
    theta(:,j,:) = (mylims(j,2)'.*exp(theta_particles(:,j,:))+mylims(j,1)')./(exp(theta_particles(:,j,:))+1);
end
log_evidence = sum(logsum_w);

%save('results_DataAnneal_unbiased_mc.mat');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [logprior] = log_prior(phi,mylims)
% Computes the log prior (takes transformed parameters as input)

sumA =  [-eye(11); eye(11); 0 0 1 0.5 0.5 0 0 0 0 0 0; 0 0 0 0 0 0 0 1 0.5 0.5 0];

sumB =  [zeros(10,1); 0.9;...
    0.5;0.3;0.99;0.5;0.5;1;0.3;0.99;0.1;0.75;0.9; 0.995;0.995];
sumB(9) = 0.2;

% Transforming back to original scale
theta = phi;
for j = 1:length(phi)
    theta(j) = (mylims(j,2)'.*exp(theta(j))+mylims(j,1)')/(exp(theta(j))+1);
end

if all(sumA*theta'<=sumB)
    logprior = sum(-phi - 2*log(1 + exp(-phi)));
else
    logprior = -inf;
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [neglogl,loglikelihood,ptseries,ntseries,likelihoods] = bege_gjrgarch_likelihood_unbiased_mc(params,data,mylims)
% Computing unbiased estimator of the likelihood of the time series under BEGE-GJR-GARCH dynamics, given observed data and model parameters
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

%Vector containing likelihood for every observation
likelihoods=zeros(length(data),1);

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

loglikelihood=loglikelihood+loglikedgam_unbiased_mc(data(1)-r_bar,ptseries(1),ntseries(1),tp,tn);
likelihoods(1)=loglikedgam_unbiased_mc(data(1)-r_bar,ptseries(1),ntseries(1),tp,tn);

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
    tmp = loglikedgam_unbiased_mc(obs,p_t,n_t,tp,tn);
    
    loglikelihood=loglikelihood+tmp;
    likelihoods(t)=tmp;
    ptseries(t)=p_t;
    ntseries(t)=n_t;
    previous_p=p_t;
    previous_n=n_t;
end
neglogl = -loglikelihood;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [loglikelihood,ptseries,ntseries] = bege_gjrgarch_likelihood_danneal_in_unbiased_mc(params,new_data,mylims,ptseries,ntseries,data_previous)
% Computing unbiased estimator of the likelihood of the "new observation" under BEGE-GJR-GARCH dynamics, given previous and new observed data and model parameters
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%COMPUTING THE LOG-LIKELIHOOD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


t1=10e-1;

previous_p = ptseries(end);
previous_n = ntseries(end);

if ((data_previous-r_bar)<0)
    p_t=max(p_bar+rho_p*previous_p+...
        phi_pn*(((data_previous-r_bar)^2)/(2*(tp^2))),t1);
    n_t=max(n_bar+rho_n*previous_n+...
        phi_nn*(((data_previous-r_bar)^2)/(2*(tn^2))),t1);
else
    p_t=max(p_bar+rho_p*previous_p+...
        phi_pp*(((data_previous-r_bar)^2)/(2*(tp^2))),t1);
    n_t=max(n_bar+rho_n*previous_n+...
        phi_np*(((data_previous-r_bar)^2)/(2*(tn^2))),t1);
end
obs = new_data -r_bar;
loglikelihood = loglikedgam_unbiased_mc(obs,p_t,n_t,tp,tn);
ptseries = [ptseries; p_t];
ntseries = [ntseries; n_t];

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function loglikedgam = loglikedgam_unbiased_mc(z,p,n,tp,tn)
% This function unbiasedly estimates the likelihood of an observation under
% the BEGE density by using Monte Carlo integration.
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


N = 10000;
wp_bar = -p*tp;
wn_bar = -n*tn;
sigma = 1/tp + 1/tn;
delta = max(wp_bar, wn_bar + z);

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
wp = gamrnd(p,tp,N,1);
wp = wp - p*tp; % de-meaned gamma draws
wp = wp-z;

x = wp + n*tn;
ind_pos = x>0;

logw = zeros(N,1);
logw(ind_pos) = -gammaln(n) - n*log(tn) + (n-1)*log(x(ind_pos)) - 1/tn*(x(ind_pos));
logw(~ind_pos) = -Inf;
loglikedgam = -log(N) + logsumexp(logw);
end









