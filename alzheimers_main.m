data = load('alzheimers/ad_data.mat');
%data.X_train = [ones(size(data.X_train,1),1) data.X_train];
%data.X_test = [ones(size(data.X_test,1),1) data.X_test];
% doesnt seem need to pad 1s ???

pars = [1e-8; 0.01; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];
aucs = zeros(12,1);
feats = zeros(12,1);

for p = 1:12
    par = pars(p);
    [w, c] = logistic_l1_train(data.X_train, data.y_train, par);
    % calculate the prediction according to the notes
    % 1 / ( 1+exp(-x^T w) )
    preds = 1 ./ (1+exp(-(data.X_test * w + c) ));
    [X,Y,T,AUC] = perfcurve(data.y_test, preds, 1);
    aucs(p) = AUC;
    w(w~=0) = 1;
    feats(p) = sum(w);
end

figure
% area(pars, aucs);
plot(pars, aucs, '--o');
xlabel('par');
ylabel('AUC');

figure
plot(pars, feats, '--o');
xlabel('par');
ylabel('number of features');