function [gamma] = Gamma(gamma_Current,gamma_Previous,N,loglike,logW_Prior)
% Gets the temperatures from likelihood annealing strategy

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gamma_Current      -  New temperature
%
% gamma_Previous     -  Old temperature
%
% N                  -  Size of population of particles
%
% loglike            -  Log-likelihood
%
% logW_Prior         -  Old samples' log-weights

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gamma              -  Difference between Effective Sample Size and the target

logW_Curr = logW_Prior + (gamma_Current - gamma_Previous)*loglike;
logW_Curr=logW_Curr-max(logW_Curr);
W=exp(logW_Curr)/sum(exp(logW_Curr));
gamma=1/sum(W.^2)-N/2;

end

