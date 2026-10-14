function p = healthy_state_path()
%HEALTHY_STATE_PATH Absolute path to the in-repo healthy_state.yaml.
%
% Both MATLAB (this helper) and C++ (--evolve-to-diagnosis flag default)
% read from resources/cpp/healthy_state.yaml so the two sides never drift.

here = fileparts(mfilename('fullpath'));
p = fullfile(here, '..', 'resources', 'cpp', 'healthy_state.yaml');
end
