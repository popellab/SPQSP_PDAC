% export_sbml.m - Export SimBiology model to SBML format
%
% Usage:
%   From pdac-build root:
%     matlab -batch "run('scripts/export_sbml.m')"
%
% Output:
%   PDAC_model.sbml in the project root directory

%% Build the model (reuses immune_oncology_model_PDAC.m logic)
immune_oncology_model_PDAC;

%% Export to SBML
output_file = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'PDAC_model.sbml');
sbmlexport(model, output_file);
fprintf('SBML model exported to: %s\n', output_file);