% Function to generate object with default model parameters
%
% Inputs: model   -- SimBiology model object
%         ID      -- T cell-antigen ID number
%         cancers -- cell array of string containing cancer names
%         P_C     -- parameter array (same length as 'cancers') of antigen concentrations per cancer cell
%         nTcells -- number of cytotoxic T cell clones in model
%
% Output: rate    -- string containing rate of antigen release from dying cancer cells


function rate = antigen_rate(model,ID,cancers,P_C,nTcells)

% Parse Inputs
P = ['P' ID];
C = cancers;

% Add Antigen Concentrations to Model
for i = 1:length(P_C)
    param = addparameter(model,[P '_' C{i}],P_C(i).Value,'ValueUnits',P_C(i).Units, 'Notes', ['Concentration of ' P ' in ' C{i}]);
    param.UserData.Source = P_C(i).Notes;
end

% Antigen-release rate is driven by R_Tcell, the model-wide cancer-killing-rate
% accumulator (declared in simbio_init.m as a repeatedAssignment). Every killing
% module appends its contribution to R_Tcell during assembly: baseline death +
% therapy + CTL killing (Teff_module), and M1 macrophage phagocytosis
% (macrophage_module). Referencing R_Tcell BY NAME
% here -- rather than string-matching the CTL killing reaction at assembly time,
% as this function used to (matched only 'k_C_Tcell_eff') -- makes antigen release:
%   (1) COMPLETE: it includes macrophage killing, which the old
%       k_C_Tcell_eff-only scan missed entirely. Those reactions are also assembled
%       AFTER antigen_module, so an assembly-time scan here could NEVER have seen
%       them regardless of the search string (the "assembly order" bug). A by-name
%       reference resolves against the fully-appended rule at simulation time.
%   (2) CONSISTENT with H_HMGB1 (APC_module), which already keys DC maturation off
%       this same R_Tcell -- so the antigen-release and DAMP arms of immunogenic
%       killing can no longer disagree (macrophage-killed cells released DAMPs but,
%       inconsistently, no antigen).
% R_Tcell already includes the baseline death + therapy term (Teff_module's append
% is '(k_Ci_death+k_Ci_therapy)*Ci + k_C_Tcell_eff*Ci'), so it is NOT added again
% here -- P_Ci*R_Tcell is the complete per-clone antigen flux.
% NOTE: R_Tcell is a single model-wide sum over ALL cancer killing, so P_Ci*R_Tcell
% is exact for the single antigen-bearing cancer clone (C1) the model uses. If
% multiple cancer types with DISTINCT antigen concentrations P_Ci were ever
% activated, antigen release would need per-cancer kill accumulators (R_Tcell_Ci)
% instead of the global R_Tcell. (mechanism-audit consensus #6, 2026-06-04)
rate = '';
for i = 1:length(C)
    if strcmp(rate,'')
        rate = sprintf('%s_%s*R_Tcell', P, C{i});        % P_Ci * R_Tcell
    else
        rate = sprintf('%s+%s_%s*R_Tcell', rate, P, C{i});
    end
end
% Map antigen ID to the new parameter naming convention
if strcmp(ID, '0')
    n_clones_param = 'n_CD4_clones';  % self-antigen clone diversity (shared CD4 pool)
elseif strcmp(ID, '1')  
    n_clones_param = 'n_CD8_clones';  % tumor antigens (renamed from n_clones_tum)
else
    error('Unknown antigen ID: %s. Expected ''0'' (self) or ''1'' (tumor)', ID);
end
rate = [n_clones_param '*(' rate ')'];
