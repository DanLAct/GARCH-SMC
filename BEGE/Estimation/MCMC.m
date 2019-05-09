function [theta, theta_transf, loglike, logprior] = MCMC(N,start_smc,cov_smc,h)
%MCMC for estimating BEGE model's parameters; 

%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%
% N                  - Total number of MCMC iterations
% start_smc          - Initial values of parameters; can be chosen from the SMC
%                      population to accelerate the convergance
% cov_smc            - Covariance of the SMC's final population of sampled
%                      parameters in transformed scale
% h                  - optimal scale from SMC used to tune the covariance

%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%
% theta              - N samples from each temperature
% theta_transf       - N transformed samples from each temperature
% loglike            - Log likelihood of the BEGE model, corresponding to the above thetas
% logprior           - Log prior of the BEGE model, corresponding to the above thetas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%loading monthly S&P stock log return
load('Data.mat');
rate_return=MonthlyReturns2018;
rate_return(isnan(rate_return))=0;%Set any empty value as zero

%Setting ranges for parameters to do transformation
mylims = zeros(11,2);
mylims(:,1) = zeros(11,1);
mylims(9,1) = -0.2;
mylims(11,1) = -0.9;
mylims(:,2) = [0.5;0.3;0.99;0.5;0.5;1;0.3;0.99;0.1;0.75;0.9];

n = 11; % dimension of theta

theta = zeros(N,n);
theta_transf = zeros(N,n);
logprior = zeros(N,1);
loglike = zeros(N,1);

theta(1,:)=start_smc; %set initial values as the ones from the final population of SMC

for j = 1:n
    theta_transf(1,j) = log((theta(1,j) - mylims(j,1))./(mylims(j,2) - theta(1,j)));
end

logprior_current=log_prior_unif(theta_transf(1,:),mylims);
logprior(1) = logprior_current;
[~, loglike_current,~,~,~] = bege_gjrgarch_likelihood(theta_transf(1,:),rate_return,mylims);
loglike(1) = loglike_current;

for i=1:N
    fprintf('Just starting with the %ith move.\n',i);
    theta_transf_prop=mvnrnd( theta_transf(i,:),h^2*cov_smc);
    logprior_prop=log_prior_unif(theta_transf_prop,mylims);
    
    if isinf(logprior_prop) ==1
        theta_transf(i+1,:) = theta_transf(i,:);
        theta(i+1,:) = theta(i,:);
        loglike(i+1) = loglike(i);
        logprior(i+1) = logprior(i);
        continue;
    end
    
    [~, loglike_prop,~,~,~] = bege_gjrgarch_likelihood(theta_transf_prop,rate_return,mylims);
    
    %Acceptance rate
    Alpha=exp(loglike_prop - loglike_current + logprior_prop - logprior_current);
    
    if rand<Alpha
        theta_transf(i+1,:)= theta_transf_prop;
        
        theta(i+1,:) = theta_transf_prop;
        for j=1:n
            theta(i+1,j) = (mylims(j,2)'.*exp(theta_transf_prop(j))+mylims(j,1)')./(exp(theta_transf_prop(j))+1);
        end
        
        logprior_current=logprior_prop;
        loglike_current=loglike_prop;
        
    else
        theta_transf(i+1,:) = theta_transf(i,:);
        theta(i+1,:) = theta(i,:);
    end
    
    loglike(i+1) = loglike_current;
    logprior(i+1) = logprior_current;
end
%save('results_bege_MCMC.mat');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [logprior] = log_prior_unif(phi,mylims)
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [neglogl,loglikelihood,ptseries,ntseries,likelihoods] = bege_gjrgarch_likelihood(params,data,mylims)
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

loglikelihood=loglikelihood+loglikedgam(data(1)-r_bar,ptseries(1),ntseries(1),tp,tn,0.001);
likelihoods(1)=loglikedgam(data(1)-r_bar,ptseries(1),ntseries(1),tp,tn,0.001);

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
    tmp = loglikedgam(obs,p_t,n_t,tp,tn,0.001);
	
    loglikelihood=loglikelihood+tmp;
    likelihoods(t)=tmp;
    ptseries(t)=p_t;
    ntseries(t)=n_t;
    previous_p=p_t;
    previous_n=n_t;
end
neglogl = -loglikelihood;

end



