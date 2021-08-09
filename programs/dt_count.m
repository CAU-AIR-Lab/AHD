function [result] = dt_count( target )
% function [result] = dt_count( target )
% result : Count matrix, result( class count, datum count, feature count )
% target : Data matrix, dt_count regards last column of target as class
% index. 
%
% Caution !! 
% you must apply grp2idx(matlab function) before
% apply dt_count to 'target'. In each feature, minimum category number start
% with 1. Also, maximum number is the number of category in each feature.
% These rules means all of category number must be ordered.
% Correct Examples : 1 2 3 4 5 6 7 8 9 10
% Incorrect Examples : 1 3 4 5 6 7 9 10 11 13

[maxrow maxcol] = size( target );
maxfeat = maxcol-1;

cc = length(unique(target(:,end))); % class count
dc = 0; % datum count
for k=1:maxfeat
    [val idx] = max( target(:,k) );
    if val > dc
        dc = val;
    end
end
fc = maxfeat;

% result( class count, datum count, feature count )
result = zeros( cc, dc, fc );
for k=1:maxrow
    for m=1:maxfeat
        result( target(k,end), target(k,m), m ) = result( target(k,end), target(k,m), m ) + 1;
    end
end
