%% Nearest Neighbor
function [res] = nn( dist_matrix, trans, tsans )
[trl tsl] = size( dist_matrix );
% trl: Train Data Length(The number of patterns)
% tsl: Test Data Length(The number of patterns)

% trans: Train Data Answer
% tsans: Test Data Answer

cc = length(unique([trans;tsans]));

correct = 0;
for k=1:tsl
    t = zeros(trl,2);
    t(:,1) = dist_matrix(:,k);
    t(:,2) = 1:trl;
    t = sortrows( t, 1 );
    
    % 만약 최소값이 여러 개라면 동일한 거리에 있는 패턴들을 모두 고려한다.
    same_idx = find(t(:,1) == t(1,1));
    same_count = length(same_idx);
    if same_count == 1
        if tsans(k) == trans(t(1,2))
            correct = correct + 1;
        end
    else
        % 최소값이 여러 개다.
        box = zeros(1,cc);
        same_answer = trans(t(same_idx,2));
        for m=1:cc
            box(1,m) = length(find(same_answer==m));
        end
        
        [val cn] = max(box);
        if tsans(k) == cn
            correct = correct + 1;
        end
    end
end

res = correct / tsl;
