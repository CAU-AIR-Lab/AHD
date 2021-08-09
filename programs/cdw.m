function [dist_matrix] = cdw( train, test )

% Get the Weight Matrix
wm = fpgd( train, 0.001, 0.0001 );

[trainrow col] = size(train);
col = col-1;
testrow = size(test, 1);

trans = train(:,end);
train(:,end) = [];
tsans = test(:,end);
test(:,end) = [];

dist_matrix = zeros(trainrow, testrow);
for k=1:trainrow
    for m=1:testrow
        dist_matrix(k,m) = ((train(k,:)-test(m,:)).^2)*(wm(trans(k),:).^2)';
    end
end
dist_matrix = sqrt(dist_matrix);
