% Aggregate Observables Module
%
% Defines derived non-species quantities that aggregate over multiple model
% species. These centralize denominators / totals that calibration target
% observable code would otherwise re-encode per target — a brittle pattern
% that drifts when species are added or renamed.
%
% Currently provided aggregates:
%   nucleated_total_T -- All nucleated cells in V_T: cancer (live + dying)
%                        + immune subsets + CAF subtypes. Mirrors the
%                        "all nucleated cells" denominator used by mIHC
%                        studies (Li et al. Cancer Cell 2022, Fig S3/S5).
%                        Used by every _pct_ calibration target.
%
% Inputs: model -- SimBiology model object with four compartments
% Outputs: model -- SimBiology model object with aggregate observables
%
% Dependencies (must run last in module assembly, after all species exist):
%   cancer_module      -- V_T.C1, V_T.C2 (resistant clone), V_T.C_x
%   Treg_module        -- V_T.Treg
%   Teff_module        -- V_T.CD8, V_T.CD8_exh
%   Th_module          -- V_T.Th, V_T.Th_exh
%   APC_module         -- V_T.cDC1, V_T.cDC2, V_T.mcDC1, V_T.mcDC2
%   MDSC_module        -- V_T.MDSC
%   macrophage_module  -- V_T.Mac_M1, V_T.Mac_M2
%   fibroblast_module  -- V_T.qPSC, V_T.iCAF, V_T.myCAF, V_T.apCAF

function model = aggregate_observables_module(model)

% nucleated_total_T: all nucleated cells in V_T
addparameter(model, 'nucleated_total_T', 0, ...
    'ValueUnits', 'cell', 'ConstantValue', false, ...
    'Notes', ['All nucleated cells in V_T (cancer + immune + CAFs). ' ...
              'Used as the denominator for _pct_ calibration targets ' ...
              'matching mIHC "% of all nucleated cells" measurements.']);

addrule(model, ['nucleated_total_T = ' ...
    'V_T.C1 + V_T.C2 + V_T.C_x' ...
    ' + V_T.CD8 + V_T.CD8_exh' ...
    ' + V_T.Th + V_T.Th_exh + V_T.Treg' ...
    ' + V_T.Tfh + V_T.B + V_T.Plasma' ...
    ' + V_T.Mac_M1 + V_T.Mac_M2' ...
    ' + V_T.MDSC' ...
    ' + V_T.cDC1 + V_T.cDC2 + V_T.mcDC1 + V_T.mcDC2' ...
    ' + V_T.qPSC + V_T.iCAF + V_T.myCAF + V_T.apCAF'], ...
    'repeatedAssignment');

end
