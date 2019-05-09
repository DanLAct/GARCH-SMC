function [gamma] = Gamma(gamma_Current,gamma_Previous,N,loglike,logW_Prior)

    logW_Curr = logW_Prior + (gamma_Current - gamma_Previous)*loglike;
    logW_Curr=logW_Curr-max(logW_Curr);
    W=exp(logW_Curr)/sum(exp(logW_Curr));
    gamma=1/sum(W.^2)-N/2;
    
end

