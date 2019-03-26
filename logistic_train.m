function [weights] = logistic_train(data, labels, epsilon, maxiter)
%
% code to train a logistic regression classifier
%
% INPUTS:
%   data    = n * (d+1) matrix withn samples and d features, where
%             column d+1 is all ones (corresponding to the intercept term)
%   labels  = n * 1 vector of class labels (taking values 0 or 1)
%   epsilon = optional argument specifying the convergence
%             criterion - if the change in the absolute difference in
%             predictions, from one iteration to the next, averaged across
%             input features, is less than epsilon, then halt
%             (if unspecified, use a default value of 1e-5)
%   maxiter = optional argument that specifies the maximum number of
%             iterations to execute (useful when debugging in case your
%             code is not converging correctly!)
%             (if unspecified can be set to 1000)
%
% OUTPUT:
%   weights = (d+1) * 1 vector of weights where the weights correspond to
%             the columns of "data"
    if nargin < 3
        epsilon = 1e-5;
        maxiter = 1000;
    elseif nargin < 4
        maxiter = 1000;    
    end
    
    eta = 0.0005;
    weights = zeros(size(data,2),1);
    
% Shaojun,
% 
% Here should be sigmoid output, instead of binary output. 
% 
% Best,
% Jiayu
    oldpred = 1 ./ (1 + exp(-data * weights) );
    for iter = 1:maxiter
        deltaE = zeros(size(data,2),1);
        % following notes gradient for 1/-1 encoding
        for i = 1:size(data,1)
            % calculate sigmoid
            sigmoid =  1/(1+exp( labels(i) * data(i,:) * weights ));
            % sum up deltaE
            deltaE = deltaE + labels(i) * sigmoid * data(i,:)';
        end
        % multiply the - 1/N term
        deltaE = - deltaE / size(data,1);
        % update weights
        weights = weights + eta * (-deltaE);
        
        newpred = 1 ./ (1 + exp(-data * weights) );
        if sum(abs(newpred-oldpred))/size(data,1) < epsilon
            break;
        end
        
        oldpred = newpred;
    end
end