function [theta, theta_particle, loglike, logprior, gamma, log_evidence] = garch_SMC_RW_LikeAnneal_parallel(N)
%SMC utilised vectorisation and parallelisation for estimating GARCH(1,1) model's
%parameters; 
%The sequence of distributions constructed by using Likelihood Annealing method

%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%
% N                  - Size of population of particles

%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%
% theta              - N samples from each temperature
% theta_particle     - N transformed samples from each temperature
% loglike            - Log likelihood of the GARCH(1,1) model, corresponding to the above thetas
% logprior           - Log prior of the GARCH(1,1) model, corresponding to the above thetas
% gamma              - The temperatures from likelihood annealing strategy
% log_evidence       - The estimate of log evidence used for model selection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%loading monthly S&P stock log return
load('Data.mat');
rate_return = MonthlyReturns2018;
rate_return(isnan(rate_return)) = 0;

%Setting ranges for parameters to do transformation
mylims = zeros(4,2);
mylims(3,1) = 0.2;
mylims(4,1) = -0.9;
mylims(:,2) = [0.3; 0.5; 0.99; 0.9];

%Starting parallel pool
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
quthpc = parcluster('local');
parpool(quthpc);
poolsize = quthpc.NumWorkers;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initialising
t = 1;
Num_Parameter = 4; % dimension of theta
log_evidence = 0;
gamma = 0; % temperatures
logw_previous = log(ones(N,1)*(1/N));
h = 0.1:0.1:1; % For choosing optimal scales and thus to reduce the number of MCMC repeats.

%Transform theta
theta = zeros(N, Num_Parameter);
theta_particle = zeros(N, Num_Parameter);
for j = 1:Num_Parameter
    theta(:,j) = unifrnd(mylims(j,1),mylims(j,2),[N,1]);
    theta_particle(:,j) = log((theta(:,j) - mylims(j,1))./(mylims(j,2) - theta(:,j)));
end

%calculating initial log prior
logprior = garch_logPrior_parallel_fv(theta_particle,mylims);


inds=1:N;
while isempty(inds)==0 %this while loop is used to eliminated any infinity value from simulation
    for j = 1:Num_Parameter
        theta(inds,j) = unifrnd(mylims(j,1),mylims(j,2),[length(inds),1]);
        theta_particle(inds,j) = log((theta(inds,j) - mylims(j,1))./(mylims(j,2) - theta(inds,j)));
    end
    logprior(inds) = garch_logPrior_parallel_fv(theta_particle(inds,:),mylims);
    inds = find(isinf(logprior));
end

%calculating initial log likelihood
[loglike,~] = garch_loglike_parallel(theta_particle, rate_return, mylims);

while gamma(t) < 1
    %Testing gamma=1
    logw = logw_previous + (1-gamma(t))*loglike(:,t);
    w = logw - max(logw); %stability
    w = exp(w);
    w = w/sum(w);
    ESS1 = 1/sum(w.^2);
    
    % Choosing next temperature
    if (ESS1 >= N/2)
        gamma(t+1) = 1;
    else
        %use bisection method to find the sequence of gamma
        fun = @(gamma_Current)Gamma(gamma_Current,gamma(t),N,loglike(:,t),logw_previous);
        interval = [gamma(t) 1];
        gamma_Current = fzero(fun,interval);
        gamma(t+1) = gamma_Current;
    end
    fprintf('The current temperature is %.3f.\n',gamma(t+1));
    
    %Substitute the value of just calculated gamma(t+1)
    logw = logw_previous +(gamma(t+1)-gamma(t))*loglike(:,t);
    log_evidence = log_evidence + logsumexp(logw);
    w = logw -max(logw);
    w = exp(w);
    w = w/sum(w);
    
    %systematic resampling - duplicating particles with more weights
    u=rand;
    indices = zeros(N,1);
    cumsum_w = cumsum(w);
    for i=1:N
        A = find(u<cumsum_w);
        indices(i) = A(1);
        u = u+1/N;
        if u>1
            u=u-1;
        end
    end
    theta_particle(:,:,t) = theta_particle(indices,:,t);
    loglike(:,t) = loglike(indices,t);
    logprior(:,t) = logprior(indices,t);
    %%% after resampling, all the particles will have equal weights %%%
    
	%(Move with MCMC Kernel)
    cov_rw = cov(theta_particle(:,:,t)); %covariance of resampled particles
    Cov_inv = cov_rw^(-1); % inverse of the above covariance
    
    % compute the mahalanobis distance form the Sample before moving
    [~,dists] = rangesearch(theta_particle(:,:,t), theta_particle(:,:,t),inf,'distance','mahalanobis','cov',cov_rw);
    dists = cell2mat(dists);
    median_dist = median(dists(:));
    
    h_ind = mod(randperm(N),length(h))'+1;
    h_all = h(h_ind);
    ESJD = zeros(N,1);
    
    %MVN RW
    parfor i=1:N
        theta_particle_prop(i,:) = mvnrnd(theta_particle(i,:,t),h_all(i)^2*cov_rw);
    end
    logprior_prop = garch_logPrior_parallel_fv(theta_particle_prop,mylims);
    
    inds = find(isinf(logprior_prop)==0);
    while isempty(inds) == 0
        %this while loop is used to eliminated any infinity value from simulation
        for j = 1:length(inds)
            theta_particle_prop(inds(j),:) = mvnrnd(theta_particle(inds(j),:,t),h_all(inds(j))^2*cov_rw);
        end
        logprior_prop(inds) = garch_logPrior_parallel_fv(theta_particle_prop(inds,:),mylims);
        inds = find(isinf(logprior_prop));
    end
    
	percore = ceil(N/poolsize);
    parfor core = 1:poolsize %Divide Sample into different 'poolsize' groups
        current_core = (percore*(core-1) + 1):min(percore*core,N);
        [loglike_prop_cell{core},~] = garch_loglike_parallel(theta_particle_prop(current_core,:),rate_return,mylims);
    end
    loglike_prop = loglike_prop_cell{1};
    for core = 2:poolsize
        loglike_prop = [loglike_prop; loglike_prop_cell{core}];
    end
    
    log_mh = gamma(t+1)*loglike_prop - gamma(t+1)*loglike(:,t) + logprior_prop - logprior(:,t);
    
	acc_probs = exp(log_mh);
    
    parfor i=1:N
        ESJD(i) = ((theta_particle(i,:,t)-theta_particle_prop(i,:))*Cov_inv*(theta_particle(i,:,t)-theta_particle_prop(i,:))')*acc_probs(i);
    end
    
    toacc_sub = find(rand(N,1) < acc_probs);
    toacc = toacc_sub;
    theta_particle(toacc,:,t)=theta_particle_prop(toacc,:);
    loglike(toacc,t)=loglike_prop(toacc);
    logprior(toacc,t)=logprior_prop(toacc);
    
    median_ESJD = accumarray(h_ind,ESJD,[],@median);%Median value of ESJD for different h indices from 1 to 10
    ind=find(median_ESJD==max(median_ESJD));
    h_opt = h(ind);
    fprintf('the scale is %f\n',h_opt);
    
    dist_move = zeros(N,1);
    belowThreshold = true;
    R_move = 0;
    
	
    % Getting number of remaining MCMC repeats
    %Performing remaining repeats
    while belowThreshold
        R_move = R_move + 1;
        
        theta_particle_prop = mvnrnd(theta_particle(:,:,t),h_opt^2*cov_rw);
		
        logprior_prop = garch_logPrior_parallel_fv(theta_particle_prop,mylims);
        
        inds = find(isinf(logprior_prop)==0);
        while isempty(inds)==0 %this while loop is used to eliminated any infinity value from simulation
            for j = 1:length(inds)
                theta_particle_prop(inds(j),:) = mvnrnd(theta_particle(inds(j),:,t),h_opt^2*cov_rw);
            end
            logprior_prop(inds) = garch_logPrior_parallel_fv(theta_particle_prop(inds,:),mylims);
            inds = find(isinf(logprior_prop));
        end
    
        percore = ceil(N/poolsize);
        parfor core = 1:poolsize %Divide Sample into different 'poolsize' groups
            current_core = (percore*(core-1) + 1):min(percore*core,N);
            [loglike_prop_cell{core},~] = garch_loglike_parallel(theta_particle_prop(current_core,:),rate_return,mylims);
        end
        loglike_prop = loglike_prop_cell{1};
        for core = 2:poolsize
            loglike_prop = [loglike_prop; loglike_prop_cell{core}];
        end
    
        
        log_mh = gamma(t+1)*loglike_prop - gamma(t+1)*loglike(:,t) + logprior_prop - logprior(:,t);
        
		acc_probs = exp(log_mh);
        
        toacc_sub = find(rand(N,1)<acc_probs);
        toacc = toacc_sub;
        
        for i=1:length(toacc)
            dist_move(toacc(i)) = dist_move(toacc(i)) + sqrt((theta_particle(toacc(i),:,t)-theta_particle_prop(toacc(i),:))*Cov_inv*(theta_particle(toacc(i),:,t)-theta_particle_prop(toacc(i),:))');
        end
        
        theta_particle(toacc,:,t) = theta_particle_prop(toacc,:);
        loglike(toacc,t) = loglike_prop(toacc);
        logprior(toacc,t) = logprior_prop(toacc);
        
        if sum(dist_move > median_dist)>=ceil(0.5*N)
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
parfor j=1:Num_Parameter
    theta(:,j,:) = (mylims(j,2)'.*exp(theta_particle(:,j,:))+mylims(j,1)')./(exp(theta_particle(:,j,:))+1);
end

delete(gcp); %shut down parallel pool

save('results_garch_LikeAnneal.mat');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [logprior] = garch_logPrior_parallel_fv(theta_particle,mylims)
% Computes the log prior (takes transformed parameters as input)

sumA = [-eye(4); eye(4); 0 1 1 0];
sumB = [-1e-9; -1e-8; -.2; 0.9; 0.3; 0.5; 0.99; 0.9; 0.9999];

[N,d] = size(theta_particle);

% Transforming back to original scale
theta = theta_particle;
for j = 1:d
    theta(:,j) = (mylims(j,2)'.*exp(theta(:,j))+mylims(j,1)')./(exp(theta(:,j))+1);
end
     
logprior = -inf*ones(N,1);
inds = find(all(sumA*theta'<=sumB*ones(1,N))==1);
logprior(inds) = sum(-theta_particle(inds,:)-2*log(1+exp(-theta_particle(inds,:))),2);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [loglikelihood,h] = garch_loglike_parallel(params,data,mylims)
% Computing the likelihood of the time series under GARCH dynamics, given observed data and model parameters
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

