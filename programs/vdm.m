function [dist_matrix] = vdm( train, test, domain_info )
% function [dist] = hd( train, test, dist_info )
% Value Difference Metric(N1)
%
% Paper Reference
% Improved Heterogeneous Distance Functions
% Journal of Artificial Intelligence Research, 6 (1997) 1-34.
% D. Randall Wilson, RANDY@AXON.CS.BYU.EDU
% Tony R. Martinez MARTINEZ@CS.BYU.EDU
%
% row: train
% col: test
% last column as class index

% 불필요한 계산을 미리 해준다.
[cc dc fc] = size(domain_info);
for k=1:fc
    for n=1:dc
        den = sum(domain_info(:,n,k));
        
        if den ~= 0
            domain_info(:,n,k) = domain_info(:,n,k) / den;
        end
    end
end

% 이제 그냥 계산만 해주면 된다.
trainrow = size(train, 1);
[testrow maxcol] = size(test);
maxfeat = maxcol-1;

dist_matrix = zeros(trainrow, testrow);
for k=1:trainrow
    for m=1:testrow
        for n=1:maxfeat
            % 같을 때에는 0이므로 계산할 필요가 없다.
            if train(k,n) ~= test(m,n)
                for a=1:cc
                    dist_matrix(k,m) = dist_matrix(k,m) + abs(domain_info(a,train(k,n),n)-domain_info(a,test(m,n),n));
                end
            end
        end
    end
end
