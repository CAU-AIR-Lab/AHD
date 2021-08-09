function [dist_matrix] = hd( train, test, dist_info )
% function [dist] = hd( train, test, dist_info )
% Hamming Distance
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
            if train(k,n) ~= test(m,n)
                dist_matrix(k,m) =  dist_matrix(k,m) + 1;
            end
        end
    end
end
