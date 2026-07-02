function params_in = generate_parameter_samples(model, params_in, n_patients, varargin)
%GENERATE_PARAMETER_SAMPLES Generate parameter samples using Latin Hypercube Sampling
%
% This function generates parameter samples for ensemble simulations using Latin
% Hypercube Sampling (LHS) from the parameter distributions defined in params_in.
%
% Inputs:
%   model      - SimBiology model object
%   params_in  - Parameter structure with distributions
%   n_patients - Number of virtual patients to generate
%   varargin   - Optional parameters:
%                'seed', value - Set random seed for reproducible sampling
%
% Outputs:
%   params_in  - Updated parameter structure with LHS samples

% Parse optional inputs
p = inputParser;
addParameter(p, 'seed', [], @isnumeric);
addParameter(p, 'cancer_type', 'PDAC', @ischar); % For prior lookup
parse(p, varargin{:});
seed_value = p.Results.seed;
cancer_type = p.Results.cancer_type;

% Set random seed if provided
if ~isempty(seed_value)
    rng(seed_value);
    fprintf('Setting random seed to %d for reproducible parameter generation\n', seed_value);
end

fprintf('Generating parameter sets for %d virtual patients...\n', n_patients);

% Create LHS fit samples between 0 and 1
lhs = lhsdesign(n_patients, length(params_in.names));

for i = 1:length(params_in.names)
    param_name = params_in.names{i};

    if strcmp(params_in.(param_name).Sampling, 'uniform')
        LB = params_in.(param_name).LowerBound;
        UB = params_in.(param_name).UpperBound;
        params_in.(param_name).LHS = LB + (UB-LB) * lhs(:,i);
        params_in.all(:,i) = params_in.(param_name).LHS;
        fprintf('  ✓ %s: sampled from uniform distribution [%.3g, %.3g]\n', param_name, LB, UB);

    elseif strcmp(params_in.(param_name).Sampling, 'loguniform')
        LB = params_in.(param_name).LowerBound;
        UB = params_in.(param_name).UpperBound;
        params_in.(param_name).LHS = 10.^( log10(LB) + (log10(UB)-log10(LB)) * lhs(:,i) );
        params_in.all(:,i) = params_in.(param_name).LHS;
        fprintf('  ✓ %s: sampled from log-uniform distribution [%.3g, %.3g]\n', param_name, LB, UB);

    elseif strcmp(params_in.(param_name).Sampling, 'normal')
        Median = params_in.(param_name).Median;
        Sigma  = params_in.(param_name).Sigma;
        params_in.(param_name).LHS = icdf('Normal',lhs(:,i),Median,Sigma);
        params_in.all(:,i) = params_in.(param_name).LHS;
        fprintf('  ✓ %s: sampled from normal distribution (μ=%.3g, σ=%.3g)\n', param_name, Median, Sigma);

    elseif strcmp(params_in.(param_name).Sampling, 'lognormal')
        Median = params_in.(param_name).Median;
        Sigma  = params_in.(param_name).Sigma;
        params_in.(param_name).LHS = icdf('Lognormal',lhs(:,i),Median,Sigma);
        params_in.all(:,i) = params_in.(param_name).LHS;
        fprintf('  ✓ %s: sampled from lognormal distribution (μ=%.3g, σ=%.3g)\n', param_name, Median, Sigma);

    elseif strcmp(params_in.(param_name).Sampling, 'binary')
        Prob = params_in.(param_name).Prob;
        Scale = params_in.(param_name).Scale;
        params_in.(param_name).LHS = zeros(n_patients,1);
        idx = find(lhs(:,i) <= Prob);
        params_in.(param_name).LHS(idx) = Scale;
        params_in.all(:,i) = params_in.(param_name).LHS;
        fprintf('  ✓ %s: sampled from binary distribution (P=%.3g, scale=%.3g)\n', param_name, Prob, Scale);

    elseif strcmp(params_in.(param_name).Sampling, 'prior')
        % NEW: Sample from empirical prior distribution
        
        % Require cancer_type and prior_source for prior sampling
        if ~isfield(params_in.(param_name), 'cancer_type')
            error('Parameter %s has Sampling=''prior'' but missing required cancer_type field', param_name);
        end
        if ~isfield(params_in.(param_name), 'prior_source')
            error('Parameter %s has Sampling=''prior'' but missing required prior_source field (''to-review'' or ''parameters'')', param_name);
        end
        
        param_cancer_type = params_in.(param_name).cancer_type;
        prior_source = params_in.(param_name).prior_source;
        
        prior_samples = load_prior_samples(param_name, param_cancer_type, n_patients, 'seed', seed_value, 'source', prior_source);
        
        if ~isempty(prior_samples)
            params_in.(param_name).LHS = prior_samples;
            params_in.all(:,i) = prior_samples;
            fprintf('  ✓ %s: sampled from empirical prior (%s/%s)\n', param_name, param_cancer_type, prior_source);
        else
            error('Prior sampling requested for %s but no prior samples available in %s/%s. Check that prior.mat or prior.npz files exist.', ...
                  param_name, param_cancer_type, prior_source);
        end
    end
end

fprintf('Parameter generation completed\n');

end
