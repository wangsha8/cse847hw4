% importing data and labels
data = importdata('spam_email/data.txt'); 
labels = importdata('spam_email/labels.txt');
% add ones two the left as bias, convert 0s to -1s
data = [ones(size(data,1),1) data];
labels(labels==0) = -1;

% split between training and test
trainingX = data(1:2000,:);
trainingY = labels(1:2000);
testingX = data(2001:4601,:);
testingY = labels(2001:4601);

% different training size
ns = [200; 500; 800; 1000; 1500; 2000];
accs = [0; 0; 0; 0; 0; 0];
for n = 1:6
    weights = logistic_train(data(1:ns(n),:), labels(1:ns(n)));
    for i = 1:size(testingX,1)
        % according to notes, if x^T w >= 0, it is positive
        % so if (x^T >= 0 and label = 1) or (x^T < 0 and label = -1), it is
        % correct prediction
        if (testingX(i,:) * weights >= 0 && testingY(i) == 1) || (testingX(i,:) * weights < 0 && testingY(i) == -1)
            accs(n) = accs(n) + 1;
        end
    end
    accs(n) = accs(n) / size(testingX,1);
end

plot(ns, accs, '--o');
xlabel('n');
ylabel('accuracy');