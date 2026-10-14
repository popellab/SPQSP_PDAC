function [prcc_results] = calculate_prcc(params_in, params_out, alpha)
% CALCULATE_PRCC - Calculate Partial Rank Correlation Coefficients
%
% Inputs:
%   params_in    - Input parameter matrix (N x k1)
%   params_out   - Output parameter matrix (N x k2)
%   alpha        - Significance level (default: 0.05)
%
% Outputs:
%   prcc_results - Struct with .prcc_values (k1 x k2), .p_values, .significant

    if nargin < 3, alpha = 0.05; end

    [n_samples, k1] = size(params_in);
    [n_samples2, k2] = size(params_out);

    % Validate dimensions
    if n_samples ~= n_samples2
        error('params_in (%d samples) and params_out (%d samples) must have same number of rows', n_samples, n_samples2);
    end

    % Initialize output matrices
    prcc_values = zeros(k1, k2);
    p_values = zeros(k1, k2);

    % For each output variable, compute partial correlations with all inputs
    % controlling only for other input variables
    for i = 1:k1
        for j = 1:k2
            % Combine inputs with current output variable
            [rho,p] = partialcorr(params_in(:,i),params_out(:,j),params_in(:,[1:i-1 , i+1:end]),"type","Spearman");
            % [rho,p] = corr(params_in(:,i),params_out(:,j),"type","Spearman");

            prcc_values(i, j) = rho;
            p_values(i, j) = p;
        end
    end

    % Package results
    prcc_results.prcc_values = prcc_values;    % k1 x k2
    prcc_results.p_values = p_values;          % k1 x k2
    prcc_results.significant = p_values < alpha;
    prcc_results.alpha = alpha;
end
