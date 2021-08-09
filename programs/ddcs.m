function [dist_info acc] = ddcs( train, domaininfo )
% function [dist_info] = ddcs( train )
% Data Driven Category Similarity

info = domaininfo;
[cc dc fc] = size(info);
maxrow = size(train, 1);

% Hamming Distance로 Setting
dist_info = ones(dc,dc,fc,cc);
for k=1:cc
    for m=1:fc
        for n=1:dc
            dist_info(n,n,m,k) = 0;
        end
    end
end

% 전체 Distance를 구해준다.
% 현재 단계에서는 Hamming Distance와 다를 바가 없다.
dist_matrix = full_dist( train, dist_info );
answer = train(:,end);

% 현재 단계에서의 정확도를 구해준다.
cur_cost = nn(dist_matrix, answer);

Terminate = false;
counter = 0;
while Terminate == false
    counter = counter + 1;    
    % 이제 전체를 돌아다니면서 숫자를 수정하면서 찾아보도록 한다.
    % 어차피 1을 0으로 바꾸는 것 밖에 없다.
    cost_matrix = zeros(dc,dc,fc,cc);
    for k=1:cc
        for m=1:fc
            for n=1:dc
                if info(k,n,m) == 0
                    % 이 도메인은 없는 도메인이다.
                    % Matlab은 정방형 매트릭스만 다루므로,
                    % 데이터에는 해당 도메인(n)이 없지만,
                    % 구조상 존재하는 것처럼 다루어질 수 있다.
                    % 이렇게 되어 있으면 탐색 시간을 낭비하게 된다.
                    continue;
                end
                
                for a=1:dc
                   if dist_info(a,n,m,k) == 1
                       dist_info(a,n,m,k) = 0;

                       % dist_matrix가 변한다.
                       diff_distance_matrix = zeros(maxrow, maxrow);
                       target_idx = find(answer(:,end)==k);
                       target_data = train(target_idx,:);
                       target_length = length(target_idx);
                       for b=1:target_length
                           if target_data(b,m) == a
                               for c=1:maxrow
                                   if train(c,m) == n
                                       diff_distance_matrix(target_idx(b),c) = 1;
                                   end
                               end
                           end
                       end

                       dist_matrix = dist_matrix - diff_distance_matrix;
                       cost_matrix(a,n,m,k) = nn( dist_matrix, answer );

                       % 복원
                       dist_info(a,n,m,k) = 1;
                       dist_matrix = dist_matrix + diff_distance_matrix;
                   end
                end
            end
        end
    end

    % cost_matrix의 값들이 모두 채워졌다.
    % 최대로 개선된 항목이 얼마인지 알아봐야 한다.
    cost_matrix = cost_matrix - cur_cost;

    max_val = -maxrow;
    max_tr_dc = -1;
    max_ts_dc = -1;
    max_fc = -1;
    max_cc = -1;

    % 찾아보자.
    for k=1:cc
        for m=1:fc
            for n=1:dc
                for a=1:dc
                    if cost_matrix(n,a,m,k) > max_val
                        max_val = cost_matrix(n,a,m,k);
                        max_tr_dc = n;
                        max_ts_dc = a;
                        max_fc = m;
                        max_cc = k;
                    end
                end
            end
        end
    end

    if max_val <= 0
        Terminate = true;
    else
        % 최대로 개선되는 대상을 찾아 1에서 0으로 바꾸어 준다.
        dist_info(max_tr_dc, max_ts_dc, max_fc, max_cc) = 0;
        cur_cost = cur_cost + max_val;
        dist_matrix = full_dist( train, dist_info );
        fprintf( 'Advanced = %5d(Class:%d), Counter = %5d, feat = %d, row = %d, col = %d\n', max_val, max_cc, counter, max_fc, max_tr_dc, max_ts_dc );
    end
end

acc = cur_cost * 100 / maxrow;

%% 전체 Distance 구하는 함수
function [dist_matrix] = full_dist( data, dist_info )
[maxrow maxcol] = size(data);
maxfeat = maxcol-1;

dist_matrix = zeros(maxrow, maxrow);
for k=1:maxrow
    for m=1:maxrow
        for n=1:maxfeat
            dist_matrix(k,m) = dist_matrix(k,m) + dist_info( data(k,n), data(m,n), n, data(k,end) );
        end
    end
end

%% LOOCV Nearest Neighbor
function [res] = nn( dist_matrix, answer )
maxrow = size( dist_matrix, 1 );
cc = length(unique(answer));

correct = 0;
for k=1:maxrow
    dist_matrix(k,k) = NaN;
    t = zeros(maxrow,2);
    t(:,1) = dist_matrix(:,k);
    t(:,2) = 1:maxrow;
    t = sortrows( t, 1 );
    
    % 만약 최소값이 여러 개라면 동일한 거리에 있는 패턴들을 모두 고려한다.
    same_idx = find(t(:,1) == t(1,1));
    same_count = length(same_idx);
    if same_count == 1
        if answer(k) == answer(t(1,2))
            correct = correct + 1;
        end
    else
        % 최소값이 여러 개다.
        box = zeros(1,cc);
        same_answer = answer(t(same_idx,2));
        for m=1:cc
            box(1,m) = length(find(same_answer==m));
        end
        
        [val cn] = max(box);
        if answer(k) == cn
            correct = correct + 1;
        end
    end
end

res = correct;

%% 10-Fold Nearest Neighbor
function [res] = foldnn( dist_matrix, answer )
maxrow = size( dist_matrix, 1 );
cc = length(unique(answer));

correct = 0;
fold = 10;
for k=1:maxrow
    dist_matrix(k,k) = NaN;
end
indices = crossvalind( 'KFold', answer, fold );

for k=1:fold
    tsrow = length(find(indices==k));
    trrow = maxrow-tsrow;
    test = (indices == k);
    train = ~test;
    tridx = find(train==true);
    tsidx = find(test==true);
    fold_dist = dist_matrix(train,test);
    
    for m=1:tsrow
        t = zeros(trrow,2);
        t(:,1) = fold_dist(:,m);
        t(:,2) = 1:trrow;
        t = sortrows( t, 1 );

        % 만약 최소값이 여러 개라면 동일한 거리에 있는 패턴들을 모두 고려한다.
        same_idx = find(t(:,1) == t(1,1));
        same_count = length(same_idx);
        if same_count == 1
            if answer(tridx(t(1,2))) == answer(tsidx(m))
                correct = correct + 1;
            end
        else
            % 최소값이 여러 개다.
            box = zeros(1,cc);
            same_answer = answer(tridx(t(same_idx,2)));
            for n=1:cc
                box(1,n) = length(find(same_answer==n));
            end

            [val cn] = max(box);
            if answer(tsidx(m)) == cn
                correct = correct + 1;
            end
        end
    end
end

res = correct;
    



























