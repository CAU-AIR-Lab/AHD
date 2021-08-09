function [cdw_acc] = IEICE20098128_CDW( alldata, iteration, perc )
%function [res] = IEICE20098128( alldata, iteration, perc )
% alldata: target data
% perc: Split�� �׽�Ʈ �������� ũ�⸦ ������� �Է�
% res: �׽�Ʈ �����Ϳ� ���� ��Ȯ��
cdw_acc = zeros(iteration,1);

for k=1:iteration
    [trainidx testidx] = crossvalind('holdout', alldata(:,end), perc/100);

    test = alldata(testidx,:);
    train = alldata(trainidx,:);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Evaluating Value Difference Metric
    fprintf( 'Evaluating Class-Dependent Weight, Stage: %d\n', k );
    
    dist_matrix = cdw( train, test );
    cdw_acc(k,1) = nn(dist_matrix, train(:,end), test(:,end));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
end

cdw_acc = cdw_acc * 100;
