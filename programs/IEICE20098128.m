function [ahd_acc hd_acc vdm_acc] = IEICE20098128( alldata, iteration, perc )
%function [res] = IEICE20098128( alldata, iteration, perc )
% alldata: target data
% perc: Split�� �׽�Ʈ �������� ũ�⸦ ������� �Է�
% res: �׽�Ʈ �����Ϳ� ���� ��Ȯ��

% ������
% 1. �־��� �����͸� ������� ���߾� �ڸ���.
% 2. ddcs�� ������ Adoptive Hamming Distance�� ���Ѵ�.
%   - �н��� Ȱ���ϴ� Classifier�� Loocv Nearest Neighbor(K=1)�̴�.
% 3. ���� Adoptive Hamming Distance�� ������� Distance Matrix�� ���Ѵ�.
% 4. Distance Matrix�� ���ؼ� nn�� ������.
% 5. �׽�Ʈ �����Ϳ� ���� ��Ȯ���� ���Ѵ�(maxacc).
% 6. 10-fold ��� 10�� ������ �� �� Max ��Ȯ���� ���Ѵ�.
ahd_acc = zeros(iteration,1);
hd_acc = zeros(iteration,1);
vdm_acc = zeros(iteration,1);

% Train �����Ϳ� Test �����͸� Split �Ͽ��� ��
% ��Ÿ�� �� �ִ� ���� ���� �ϳ���
% Train �����Ϳ����� �����µ�
% Test �����Ϳ����� �����ϴ� �������� �߻��� �� �ִٴ� ���̴�.
% �̷��� ������ �ذ��ϱ� ���ؼ� ��ü �����Ϳ��� ������ ������ ���´�.
% domain_info ������ �������� ������ ��µ����� Ȱ��ȴ�.
% (Aoptive Hamming Distance ���� �ٸ� �޼ҵ忡����
%  Ȱ���� �� �ִ� ���� �Լ��̴�)
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
    % Value Difference Metric���� Ȱ���� ��Ʈ����
    % �����Ͱ� Split �Ǿ����Ƿ�, Train �����Ϳ� ���Ե� ��������
    % ������ �޶����� �ȴ�.
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
