function [ahd_acc hd_acc vdm_acc] = IEICE20098128( alldata, iteration, perc )
%function [res] = IEICE20098128( alldata, iteration, perc )
% alldata: target data
% perc: Split될 테스트 데이터의 크기를 백분율로 입력
% res: 테스트 데이터에 대한 정확도

% 실험은
% 1. 주어진 데이터를 백분율에 맞추어 자른다.
% 2. ddcs를 돌려서 Adoptive Hamming Distance를 구한다.
%   - 학습에 활용하는 Classifier는 Loocv Nearest Neighbor(K=1)이다.
% 3. 구한 Adoptive Hamming Distance를 기반으로 Distance Matrix를 구한다.
% 4. Distance Matrix를 통해서 nn를 돌린다.
% 5. 테스트 데이터에 대한 정확도를 구한다(maxacc).
% 6. 10-fold 라면 10번 돌려서 그 중 Max 정확도를 구한다.
ahd_acc = zeros(iteration,1);
hd_acc = zeros(iteration,1);
vdm_acc = zeros(iteration,1);

% Train 데이터와 Test 데이터를 Split 하였을 때
% 나타날 수 있는 문제 중의 하나는
% Train 데이터에서는 없었는데
% Test 데이터에서는 존재하는 도메인이 발생할 수 있다는 점이다.
% 이러한 문제를 해결하기 위해서 전체 데이터에서 도메인 개수만 얻어온다.
% domain_info 변수는 도메인의 개수를 얻는데에만 활용된다.
% (Aoptive Hamming Distance 외의 다른 메소드에서도
%  활용할 수 있는 범용 함수이다)
domain_info = dt_count(alldata);

for k=1:iteration
    [trainidx testidx] = crossvalind('holdout', alldata(:,end), perc/100);

    test = alldata(testidx,:);
    train = alldata(trainidx,:);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Evaluating Adoptive Hamming Distance
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    fprintf( 'Evaluating Adoptive Hamming Distance, Stage: %d\n', k );
    dist_info = ddcs( train, domain_info );
    dist_matrix = ahd( train, test, dist_info );
    ahd_acc(k,1) = nn(dist_matrix, train(:,end), test(:,end));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % Evaluating Hamming Distance
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    fprintf( 'Evaluating Hamming Distance, Stage: %d\n', k );
    dist_matrix = hd( train, test );
    hd_acc(k,1) = nn(dist_matrix, train(:,end), test(:,end));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Evaluating Value Difference Metric
    fprintf( 'Evaluating Value Difference Metric, Stage: %d\n', k );
    
    % Initialize Dummy Matrix
    % Value Difference Metric에서 활용할 매트릭스
    % 데이터가 Split 되었으므로, Train 데이터에 포함된 도메인의
    % 개수가 달라지게 된다.
    dummy_info = zeros(size(domain_info));
    
    fold_info = dt_count( train );
    [tcc tdc tfc] = size(fold_info);
    for icc=1:tcc
        for idc=1:tdc
            for ifc=1:tfc
                dummy_info(icc,idc,ifc) = fold_info(icc,idc,ifc);
            end
        end
    end
    dist_matrix = vdm( train, test, dummy_info );
    vdm_acc(k,1) = nn(dist_matrix, train(:,end), test(:,end));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
end

ahd_acc = ahd_acc * 100;
hd_acc = hd_acc * 100;
vdm_acc = vdm_acc * 100;
