function loglikedgam = loglikedgam_parallel(z,p,n,tp,tn,zint)
% This function numerically estimates the likelihood
% of an observation under the BEGE density.
% This is done by numerically evaluating the cdf of the 
% distribution at two points just above and just below the
% observed data point, and then taking the finite difference approximation
% to the derivative (i.e., the pdf is the derivative of the cdf).
%
% Input:
% z - the point at which the pdf is evaluated
% p - good environment shape parameter
% n - bad environment shape parameter
% tp - good environment scale parameter
% tn - bad environment scale parameter
% zint - the finite difference approximation interval. In the application to the monthly US aggregate stock
% market returns, we have found the value of 0.001 (and smaller)
% reasonable.
%
% Output:
% loglikedgam - the loglikelihood of the observations (log of the pdf
% value)
%
%
% If you are using this code, please cite the following paper:
%
% Bekaert, G., Engstrom, E., and A. Ermolov, 2015, Bad environments, good
% environments: A non-Gaussian asymmetric volatility model, Journal of Econometrics, 1 (186), pp. 258–275. 
%
% The code is absolutely free to use, but we are not responsible for any
% damage it can create. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%The original codes described above have been vectorised to speed up
%computation. So the relevant input parameters, p,n,tp and tn, are vectors. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = length(z);
ii = 1;
for z = [(z-zint/2) (z+zint/2)]

% define grid of points for pt over which to
% integrate
pmin  = -p.*tp + 1e-4;
pmax  = 10*sqrt(p).*tp;
np    = 100;
pint  = (pmax-pmin)/(np+1);
pgrid = linspaceNDim(pmin, pmax, np+2)';

% for each pt, nt must be equal to
% pt - z;
ngrid = pgrid-ones(102,1)*pint' - ones(102,1)*z';

cp = cdf('gam', pgrid + ones(102,1)*(p.*tp)', ones(102,1)*p', ones(102,1)*tp');
pp = (diff([zeros(1,N); cp]));

cn = 1 - cdf('gam', ngrid + ones(102,1)*(n.*tn)', ones(102,1)*n',ones(102,1)*tn');

cz(ii,:) = sum(cn.*pp)';
ii = ii + 1;
end

% finite difference approximation
pz = diff(cz)/zint;

loglikedgam = log(min(max(pz,1e-20),1e20))';
end