function [res] = cattonbin( alldata )

answer = alldata(:,end);
alldata(:,end) = [];
[row col] = size(alldata);
res = [];

for k=1:col
    num = length(unique( alldata(:,k) ));
    tres = ones(row,num);
    for m=1:num
        tres(find(alldata(:,k)==m),m) = 2;
    end
    
    res = [res tres];
end
res = [res answer];
