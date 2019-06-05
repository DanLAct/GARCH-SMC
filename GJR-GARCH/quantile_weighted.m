function [quantile] = quantile_weighted(samples,p,weights)
% Gets quantiles for a weighted sample

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% samples        -  The sample particles (that have associated weights)
%
% p              -  Desired quantile
%
% weights        -  Weights associated with samples

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% quantile       -  The sample associated with desired quantile p.
% load('Test_rate_returnPredict.mat');
% load('Test_W.mat');
% samples=rate_returnPredict(:,1);
% p=0.975;
% weights=W(:,end-110+1);

[N, d] = size(samples);
quantile = zeros(1,d);

if length(weights)~=N
    error('The samples and weights should be of the same size!');
end

%Different from normal since bottom isn't 0 and top isn't 1...

for j=1:d
    [sorted, order] = sort(samples(:,j));%'sorted'-the whole sorted samples;'order'-corresponding Indices of the sample
    cumsum_w = weights(order);
    cumsum_w = cumsum(cumsum_w);
    lower = find(cumsum_w<p+10^(-15),1,'last');
    if isempty(lower)==1
        quantile(j)=p/cumsum_w(1)*sorted(1);
        continue
    end
    %     lower = find(cumsum_w<p,1,'last');
    upper = lower + 1;
    quantile(j) = sorted(lower) + (p - cumsum_w(lower))/(cumsum_w(upper) - cumsum(lower))*(sorted(upper) - sorted(lower)); %Linear interpolation
end

end
