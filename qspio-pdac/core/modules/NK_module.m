% NK Cell Module
%
% NOT ASSEMBLED: this module is not called from
% scripts/immune_oncology_model_PDAC.m in the current build. Its parameters
% (k_IFNg_NK_sec, NK recruitment/killing rates) are therefore NOT defined
% in parameters/parameters_PDAC.m and would raise a struct-field error if
% the module were invoked without restoring them. Consequently the NK arm
% of M2->M1 repolarization (via NK-derived IFNg) is absent from the PDAC
% model; all IFNg is T-cell-derived. The k_IFNg_NK_sec string-match in
% metabolism_module.m:201 is a conditional over reaction rates, so it is
% silently a no-op while NK_module is off. Reactivate by (a) restoring the
% missing parameter definitions in parameters_PDAC.m, (b) adding an
% NK_module(model, params, cancer_types) call in the main assembly script,
% and (c) adding the params to pdac_priors.csv if they should be varied
% in SBI.
%
% Models NK cell recruitment, innate killing of tumor cells, and IFNg production.
% NK cells provide antigen-independent innate killing alongside CD8 T cells.
% NK function is suppressed by IL-10, TGFb, and MDSCs (ArgI/NO).
% NK-derived IFNg feeds into M2->M1 macrophage polarization.
%
% Inputs: model        -- SimBiology model object with four compartments
%         params       -- object containing model parameter Values, Units, and Notes
%         cancer_types -- cell array of cancer type names (e.g. {'C1'})
%
% Outputs: model -- SimBiology model object with new NK cell module

function model = NK_module(model, params, cancer_types)

% Add Species
NK = addspecies(model.Compartment(3), 'NK', 0, 'InitialAmountUnits', 'cell');
    set(NK, 'Notes', 'Number of NK cells in the tumor compartment');
NK_P = addspecies(model.Compartment(2), 'NK', params.NK_P_steady.Value, 'InitialAmountUnits', 'cell');
    set(NK_P, 'Notes', 'Number of NK cells in the peripheral compartment');

% Add Parameters
p = addparameter(model, 'k_NK_rec', params.k_NK_rec.Value, 'ValueUnits', params.k_NK_rec.Units, 'Notes', 'NK cell recruitment rate to tumor');
p.UserData.Source = params.k_NK_rec.Notes;
    p.UserData.derivedFrom = params.k_NK_rec.derivedFrom;
p = addparameter(model, 'k_NK_death', params.k_NK_death.Value, 'ValueUnits', params.k_NK_death.Units, 'Notes', 'NK cell death/turnover rate');
p.UserData.Source = params.k_NK_death.Notes;
    p.UserData.derivedFrom = params.k_NK_death.derivedFrom;
p = addparameter(model, 'k_NK_kill', params.k_NK_kill.Value, 'ValueUnits', params.k_NK_kill.Units, 'Notes', 'NK cell killing rate of tumor cells');
p.UserData.Source = params.k_NK_kill.Notes;
    p.UserData.derivedFrom = params.k_NK_kill.derivedFrom;
p = addparameter(model, 'K_NK_C', params.K_NK_C.Value, 'ValueUnits', params.K_NK_C.Units, 'Notes', 'NK-tumor association coefficient');
p.UserData.Source = params.K_NK_C.Notes;
    p.UserData.derivedFrom = params.K_NK_C.derivedFrom;
p = addparameter(model, 'k_IFNg_NK_sec', params.k_IFNg_NK_sec.Value, 'ValueUnits', params.k_IFNg_NK_sec.Units, 'Notes', 'IFNg secretion rate by NK cells');
p.UserData.Source = params.k_IFNg_NK_sec.Notes;
    p.UserData.derivedFrom = params.k_IFNg_NK_sec.derivedFrom;
p = addparameter(model, 'IL10_50_NK', params.IL10_50_NK.Value, 'ValueUnits', params.IL10_50_NK.Units, 'Notes', 'IL-10 EC50 for NK suppression');
p.UserData.Source = params.IL10_50_NK.Notes;
    p.UserData.derivedFrom = params.IL10_50_NK.derivedFrom;
p = addparameter(model, 'TGFb_50_NK', params.TGFb_50_NK.Value, 'ValueUnits', params.TGFb_50_NK.Units, 'Notes', 'TGFb EC50 for NK suppression');
p.UserData.Source = params.TGFb_50_NK.Notes;
    p.UserData.derivedFrom = params.TGFb_50_NK.derivedFrom;
p = addparameter(model, 'IL12_50_NK', params.IL12_50_NK.Value, 'ValueUnits', params.IL12_50_NK.Units, 'Notes', 'IL-12 EC50 for NK activation');
p.UserData.Source = params.IL12_50_NK.Notes;
    p.UserData.derivedFrom = params.IL12_50_NK.derivedFrom;
p = addparameter(model, 'q_NK_P_in', params.q_NK_P_in.Value, 'ValueUnits', params.q_NK_P_in.Units, 'Notes', 'NK peripheral homeostasis rate');
p.UserData.Source = params.q_NK_P_in.Notes;
    p.UserData.derivedFrom = params.q_NK_P_in.derivedFrom;
p = addparameter(model, 'NK_P_steady', params.NK_P_steady.Value, 'ValueUnits', params.NK_P_steady.Units, 'Notes', 'Peripheral NK steady-state count');
p.UserData.Source = params.NK_P_steady.Notes;
    p.UserData.derivedFrom = params.NK_P_steady.derivedFrom;
p = addparameter(model, 'vol_NK', params.vol_NK.Value, 'ValueUnits', params.vol_NK.Units, 'Notes', 'Volume of a single NK cell');
p.UserData.Source = params.vol_NK.Notes;
    p.UserData.derivedFrom = params.vol_NK.derivedFrom;

% Hill Functions (repeated assignment rules)
H_IL10_NK = addparameter(model, 'H_IL10_NK', 0, 'ValueUnits', 'dimensionless', 'ConstantValue', false, ...
    'Notes', 'Hill function for IL-10 suppression of NK function');
H_TGFb_NK = addparameter(model, 'H_TGFb_NK', 0, 'ValueUnits', 'dimensionless', 'ConstantValue', false, ...
    'Notes', 'Hill function for TGFb suppression of NK function');
H_IL12_NK = addparameter(model, 'H_IL12_NK', 0, 'ValueUnits', 'dimensionless', 'ConstantValue', false, ...
    'Notes', 'Hill function for IL-12 activation of NK function');

addrule(model, 'H_IL10_NK = V_T.IL10/(IL10_50_NK + V_T.IL10)', 'repeatedAssignment');
addrule(model, 'H_TGFb_NK = V_T.TGFb/(TGFb_50_NK + V_T.TGFb)', 'repeatedAssignment');
addrule(model, 'H_IL12_NK = V_T.IL12/(IL12_50_NK + V_T.IL12)', 'repeatedAssignment');

% Reactions
% Peripheral NK homeostasis (constant production + first-order turnover -> steady state at NK_P_steady)
reaction = addreaction(model, 'null -> V_P.NK');
    set(reaction, 'ReactionRate', 'q_NK_P_in*NK_P_steady');
    set(reaction, 'Notes', 'NK cell production in peripheral compartment');
reaction = addreaction(model, 'V_P.NK -> null');
    set(reaction, 'ReactionRate', 'q_NK_P_in*V_P.NK');
    set(reaction, 'Notes', 'NK cell turnover in peripheral compartment');

% NK recruitment to tumor (scaled by peripheral pool fraction and CXCL12 exclusion)
reaction = addreaction(model, 'null -> V_T.NK');
    set(reaction, 'ReactionRate', 'k_NK_rec*V_T*(V_P.NK/(NK_P_steady+cell))*(1-H_CXCL12_Texcl)');
    set(reaction, 'Notes', 'NK cell recruitment to tumor (scaled by peripheral pool, reduced by CXCL12 exclusion)');

% NK death in tumor
reaction = addreaction(model, 'V_T.NK -> null');
    set(reaction, 'ReactionRate', 'k_NK_death*V_T.NK');
    set(reaction, 'Notes', 'NK cell death in tumor');

% NK clearance upon tumor eradication
reaction = addreaction(model, 'V_T.NK -> null');
    set(reaction, 'ReactionRate', 'k_cell_clear*V_T.NK*(K_C2_recruit/(C_total^2 + K_C2_recruit))');
    set(reaction, 'Notes', 'NK cell clearance upon tumor eradication');

% NK-derived IFNg secretion (suppressed by TGFb, enhanced by IL-12)
% IL-12 from cDC1 and M1 macrophages is a potent NK activator for IFNg production
reaction = addreaction(model, 'null -> V_T.IFNg');
    set(reaction, 'ReactionRate', 'k_IFNg_NK_sec*V_T.NK*(1-H_TGFb_NK)*(1+H_IL12_NK)');
    set(reaction, 'Notes', 'IFNg secretion by NK cells (suppressed by TGFb, enhanced by IL-12)');

% XCL1 axis: NK -> XCL1 -> cDC1 recruitment
% XCL1 species
addspecies(model.Compartment(3), 'XCL1', 0, 'InitialAmountUnits', 'nanomolarity');

% XCL1 parameters
p = addparameter(model, 'k_XCL1_NK', params.k_XCL1_NK.Value, 'ValueUnits', params.k_XCL1_NK.Units, 'Notes', 'XCL1 secretion rate by NK cells');
p.UserData.Source = params.k_XCL1_NK.Notes;
    p.UserData.derivedFrom = params.k_XCL1_NK.derivedFrom;
p = addparameter(model, 'k_XCL1_deg', params.k_XCL1_deg.Value, 'ValueUnits', params.k_XCL1_deg.Units, 'Notes', 'XCL1 degradation rate');
p.UserData.Source = params.k_XCL1_deg.Notes;
    p.UserData.derivedFrom = params.k_XCL1_deg.derivedFrom;
p = addparameter(model, 'XCL1_50_cDC1', params.XCL1_50_cDC1.Value, 'ValueUnits', params.XCL1_50_cDC1.Units, 'Notes', 'XCL1 EC50 for cDC1 recruitment');
p.UserData.Source = params.XCL1_50_cDC1.Notes;
    p.UserData.derivedFrom = params.XCL1_50_cDC1.derivedFrom;

% H_XCL1_cDC1 rule (overwrites simbio_init default of 0)
addrule(model, 'H_XCL1_cDC1 = V_T.XCL1/(XCL1_50_cDC1 + V_T.XCL1)', 'repeatedAssignment');

% XCL1 secretion by NK (suppressed by TGFb, enhanced by IL-12)
% Completes loop: IL-12 -> NK activation -> XCL1 -> cDC1 recruitment -> IL-12
reaction = addreaction(model, 'null -> V_T.XCL1');
    set(reaction, 'ReactionRate', 'k_XCL1_NK*V_T.NK*(1-H_TGFb_NK)*(1+H_IL12_NK)');
    set(reaction, 'Notes', 'XCL1 secretion by NK cells (suppressed by TGFb, enhanced by IL-12)');

% XCL1 degradation
reaction = addreaction(model, 'V_T.XCL1 -> null');
    set(reaction, 'ReactionRate', 'k_XCL1_deg*V_T.XCL1');
    set(reaction, 'Notes', 'XCL1 degradation');

% Get Model Rules for Updating
model_rules = get(model, 'Rules');

% Update Tumor Volume (Rule 1) - add NK cell contribution
volume_rule = model_rules(1);
rule = get(volume_rule, 'Rule');
set(volume_rule, 'Rule', [rule '+V_T.NK*vol_NK/Ve_T']);

% NK Killing of Cancer Cells and Antigen Release Update
if (exist('cancer_types', 'var'))
    for i = 1:length(cancer_types)
        % NK killing (suppressed by IL-10, TGFb, MDSCs; enhanced by IL-12)
        reaction = addreaction(model, ['V_T.' cancer_types{i} ' -> V_T.C_x']);
            set(reaction, 'ReactionRate', ['k_NK_kill*V_T.' cancer_types{i} ...
                '*V_T.NK/(V_T.NK+K_NK_C*C_total+cell)*(1-H_IL10_NK)*(1-H_TGFb_NK)*(1-H_MDSC)*(1+H_IL12_NK)']);
            set(reaction, 'Notes', 'NK cell-mediated killing of cancer cell');
        % Update antigen release rate (Rule 5)
        rule = get(model_rules(5), 'Rule');
        set(model_rules(5), 'Rule', [rule, '+k_NK_kill*V_T.' cancer_types{i} ...
            '*V_T.NK/(V_T.NK+K_NK_C*C_total+cell)*(1-H_IL10_NK)*(1-H_TGFb_NK)*(1-H_MDSC)*(1+H_IL12_NK)']);
    end
end
