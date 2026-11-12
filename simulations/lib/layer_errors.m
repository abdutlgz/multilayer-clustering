function [total,activeErr,nonactiveErr]=layer_errors(pred,truth,M,active)
% Fraction of misclassified layers, minimized over the M! relabelings of the groups.
%   total        : over all layers (the "total error")
%   activeErr    : over the active layers (the "clustering error"), NaN if none
%   nonactiveErr : over the non-active layers (the "classification error"), NaN if none
pp=perms(1:M); best=inf;
for j=1:size(pp,1)
    mapping=pp(j,:); wrong=pred(:)~=reshape(mapping(truth),[],1);
    if sum(wrong)<best, best=sum(wrong); chosen=wrong; end
end
total=mean(chosen); activeErr=NaN; nonactiveErr=NaN;
if ~isempty(active), activeErr=mean(chosen(active)); end
inactive=setdiff(1:numel(truth),active);
if ~isempty(inactive), nonactiveErr=mean(chosen(inactive)); end
end
