function [wm] = fpgd( train, lr, e )

% A Class-Dependent Weighted Dissimilarity Measure for Nearest Neighbor
% Classification Problems
%
% Roberto Paredes and Enrique Vidal
% Pattern Recognition Letters, 2000
%
% Written by Jae-sung, Lee 2009
% jslee.cau@gmail.com

% Fractional Programming Gradient Descent(FPGD algorithm)
% r         f(m)/g(m)
% r1        next step of r
% r2        temporal r, after learning based on learning rate lr
% lr        learning rate
% e         the precision of the minimum required to assess convergence
%
% In the letter, author is set the parameter lr as 0.001
% but eg and er is not mentioned specifically
%
% In this source code, 'train' variable means that 'S' in the letter.
% also, the last column of 'train' matrix is regarded as class index
% In addition, CDW does not cover the multi-nary data sets
% It needs some preprocessing steps, before this algorithm is applied.
% In the letter, author preprocessed the categorical feature to
% n binary features(if the categorical feature represented as 3 domains,
% then n binary features are made like a below example)
% 
% Categorical Feature            n-binary feautre
%        A                          1   0   0
%        A                          1   0   0
%        A                          1   0   0
%        A                          1   0   0
%        B                          0   1   0
%        B                          0   1   0
%        B                          0   1   0
%        C                          0   0   1
%        C                          0   0   1
%        C                          0   0   1
%        C                          0   0   1
%
% CAUTION!!
% Before you apply this code to your own data set
% Please check below notices
% 1. Class index must take the form of ordered integer
%    if the data set is separated 3 classes, then class index must
%    represent 1, 2 and 3.
% 2. All of domains in each feature are represented as same as notice 1.
% 3. If you want some more information, then you can get these information
%    on MATLAB HELP: typing 'help grp2idx'

% Check the size of train
answer = train(:,end);
train(:,end) = [];
[maxrow maxfeat] = size(train);
cc = length(unique(answer));

% Initialize weight matrix
wm = ones(cc,maxfeat);
wm1 = ones(cc,maxfeat);

r1 = inf;
r = inter( train, answer, wm ) / outer( train, answer, wm );
while r1-r > e
    r1 = r;
    r2 = inf;
    while r2-r > e
        r2 = r;
        wm1 = wm;
        for k=1:maxrow
            hi = answer(k);
            
            [onn_idx onn_dt] = outerclass_nn( train, answer, k, wm );
            onn = train(onn_idx,:);
            hk = answer(onn_idx);
            
            [inn_idx inn_dt] = interclass_nn( train, answer, k, wm );
            inn = train(inn_idx,:);
            for m=1:maxfeat
                wm1(hi,m) = wm1(hi,m) - ( lr*wm(hi,m)*((inn(m)-train(k,m))^2) ) / inn_dt;
                wm1(hk,m) = wm1(hk,m) + ( r1*lr*wm(hk,m)*((onn(m)-train(k,m))^2) ) / onn_dt;
            end
        end
        wm = wm1;
        r = inter( train, answer, wm ) / outer( train, answer, wm );
    end
end

function [idx dt] = interclass_nn( train, answer, targetidx, wm )
tc = answer(targetidx);
tdata = train(find(answer==tc),:);
tidx = find(answer==tc);
[row col] = size(tdata);
tp = train(targetidx,:);
td = zeros(row,1);
for k=1:row
    td(k) = td(k) + ((tp(1,:)-tdata(k,:)).^2)*(wm(tc,:).^2)';
end
td = sqrt(td);
td(find(td==0)) = NaN;
[minval minidx] = min(td);
dt = minval;
idx = tidx(minidx);

function [idx dt] = outerclass_nn( train, answer, targetidx, wm )
tc = answer(targetidx);
ntdata = train(find(answer~=tc),:);
ntidx = find(answer~=tc);
[row col] = size(ntdata);
tp = train(targetidx,:);
ntd = zeros(row,1);
for k=1:row
    ntd(k) = ntd(k) + ((tp(1,:)-ntdata(k,:)).^2)*(wm(tc,:).^2)';
end
ntd = sqrt(ntd);
ntd(find(ntd==0)) = NaN;
[minval minidx] = min(ntd);
dt = minval;
idx = ntidx(minidx);

function [val] = inter( train, answer, wm )

% This function is represented as f(M) in the letter
cc = length(unique(answer));
val = 0;
for k=1:cc
    tdata = train(find(answer==k),:);
    % 거리를 구하자
    [row col] = size(tdata);
    td = zeros(row,row);
    for m=1:row
        for n=m+1:row
            td(m,n) = ((tdata(m,:)-tdata(n,:)).^2)*(wm(k,:).^2)';
            td(n,m) = td(m,n);
        end
        td(m,m) = NaN;
    end

    td = sqrt(td);
    val = val + sum(min(td));
end

function [val] = outer( train, answer, wm )

% This function is represented as g(M) in the letter
cc = length(unique(answer));
val = 0;
for k=1:cc
    tdata = train(find(answer==k),:);
    [trow col] = size(tdata);
    
    ntdata = train(find(answer~=k),:);
    ntrow = size(ntdata, 1);
    
    td = zeros(ntrow, trow);
    for m=1:ntrow
        for n=1:trow
            td(m,n) = ((tdata(n,:)-ntdata(m,:)).^2)*(wm(k,:).^2)';
        end
    end
    
    td = sqrt(td);
    val = val + sum(min(td));
end

















