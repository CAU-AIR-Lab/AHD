function [dist_matrix] = ahd( train, test, dist_info )
% function [dist] = adh( train, test, dist_info )
% Adoptive Hamming Distance
% row: train
% col: test
% last column as class index

trainrow = size(train, 1);
[testrow maxcol] = size(test);
maxfeat = maxcol-1;

dist_matrix = zeros(trainrow, testrow);
for k=1:trainrow
    for m=1:testrow
        for n=1:maxfeat
            dist_matrix(k,m) = dist_matrix(k,m) + dist_info( train(k,n), test(m,n), n, train(k,end) );
        end
    end
end
