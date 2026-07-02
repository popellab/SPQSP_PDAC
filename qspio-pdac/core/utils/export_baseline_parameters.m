function export_baseline_parameters(model_name, output_csv)
%EXPORT_BASELINE_PARAMETERS Export baseline parameter values from model
%
% Instantiates the model and exports parameter values to a CSV file
% for comparison with LLM-estimated ranges. The model uses its internally
% defined parameters.
%
% Inputs:
%   model_name  - Name of model function (e.g., 'immune_oncology_model_PDAC')
%   output_csv  - Output CSV file path
%
% Output CSV format:
%   parameter_id,baseline_value,units
%
% Example:
%   export_baseline_parameters('immune_oncology_model_PDAC', 'temp_baseline_params.csv');

% Run model script (uses internally defined parameters)
fprintf('Running model script: %s\n', model_name);

% Clear workspace to avoid conflicts
clear model

% Run the script - it should create a 'model' variable
eval(model_name);

% Check if model was created
if ~exist('model', 'var')
    error('Model script did not create a ''model'' variable');
end

% Extract parameter values
model_params = get(model, 'Parameters');

fprintf('Exporting %d baseline parameter values...\n', length(model_params));

% Write to CSV
fid = fopen(output_csv, 'w');
fprintf(fid, 'parameter_id,baseline_value,units\n');

for i = 1:length(model_params)
    param = model_params(i);
    fprintf(fid, '%s,%.10e,%s\n', param.Name, param.Value, param.ValueUnits);
end

fclose(fid);

fprintf('✓ Exported baseline parameters to: %s\n', output_csv);

end
