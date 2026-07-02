% Hypoxia-HIF-VEGF Feedback Module
%
% Models tumor hypoxia, HIF-1alpha signaling, and downstream effects on
% VEGF amplification, PD-L1 upregulation, and M2 macrophage polarization.
%
% PDAC is among the most hypoxic solid tumors (median pO2 ~2.7 mmHg vs
% ~50 mmHg normal tissue). This module creates a positive feedback loop:
%   hypoxia -> HIF -> VEGF -> aberrant vasculature -> more hypoxia
% and connects it to immunosuppressive circuits (PD-L1, M2 polarization).
%
% Requires: cancer_module (K, VEGF, C_total), Teff_module (T_total),
%           macrophage_module (M_total, M1->M2 reaction),
%           checkpoint_module (syn_CD8_C1 synapse)
% Forward-references: aPSC_total (fibroblast_module_enhanced)
%
% Inputs: model  -- SimBiology model object
%         params -- object containing model parameter Values, Units, and Notes
%
% Outputs: model -- SimBiology model object with hypoxia-HIF-VEGF feedback

function model = hypoxia_module(model, params)

%% Add Parameters
p = addparameter(model,'pO2_normal',params.pO2_normal.Value,'ValueUnits',params.pO2_normal.Units, 'Notes', 'Normal tissue pO2 (mmHg)');
p.UserData.Source = params.pO2_normal.Notes;
    p.UserData.derivedFrom = params.pO2_normal.derivedFrom;

p = addparameter(model,'k_O2_demand',params.k_O2_demand.Value,'ValueUnits',params.k_O2_demand.Units, 'Notes', 'O2 demand coefficient per metabolically active cell');
p.UserData.Source = params.k_O2_demand.Notes;
    p.UserData.derivedFrom = params.k_O2_demand.derivedFrom;

p = addparameter(model,'VEGF_norm_50',params.VEGF_norm_50.Value,'ValueUnits',params.VEGF_norm_50.Units, 'Notes', 'VEGF level for half-max vascular aberration');
p.UserData.Source = params.VEGF_norm_50.Notes;
    p.UserData.derivedFrom = params.VEGF_norm_50.derivedFrom;

p = addparameter(model,'pO2_50',params.pO2_50.Value,'ValueUnits',params.pO2_50.Units, 'Notes', 'pO2 for half-max HIF activation (mmHg)');
p.UserData.Source = params.pO2_50.Notes;
    p.UserData.derivedFrom = params.pO2_50.derivedFrom;

p = addparameter(model,'n_HIF',params.n_HIF.Value,'ValueUnits',params.n_HIF.Units, 'Notes', 'Hill coefficient for HIF activation');
p.UserData.Source = params.n_HIF.Notes;
    p.UserData.derivedFrom = params.n_HIF.derivedFrom;

p = addparameter(model,'k_VEGF_HIF',params.k_VEGF_HIF.Value,'ValueUnits',params.k_VEGF_HIF.Units, 'Notes', 'HIF-driven VEGF secretion rate per cancer cell');
p.UserData.Source = params.k_VEGF_HIF.Notes;
    p.UserData.derivedFrom = params.k_VEGF_HIF.derivedFrom;

p = addparameter(model,'k_PDL1_HIF',params.k_PDL1_HIF.Value,'ValueUnits',params.k_PDL1_HIF.Units, 'Notes', 'HIF-driven PD-L1 synthesis rate on cancer cells');
p.UserData.Source = params.k_PDL1_HIF.Notes;
    p.UserData.derivedFrom = params.k_PDL1_HIF.derivedFrom;

p = addparameter(model,'r_PDL1_HIF',params.r_PDL1_HIF.Value,'ValueUnits',params.r_PDL1_HIF.Units, 'Notes', 'Max fold-increase PD-L1 from HIF');
p.UserData.Source = params.r_PDL1_HIF.Notes;
    p.UserData.derivedFrom = params.r_PDL1_HIF.derivedFrom;

%% Add Algebraic Parameters (quasi-steady-state, not ODE species)
% pO2 and HIF equilibrate on seconds-to-hours timescales vs the model's daily timestep

pO2 = addparameter(model,'pO2',0,'ValueUnits','millimeter_mercury','ConstantValue',false, ...
    'Notes', 'Tumor oxygen partial pressure (mmHg). Quasi-steady-state.');

HIF = addparameter(model,'HIF',0,'ValueUnits','dimensionless','ConstantValue',false, ...
    'Notes', 'HIF-1alpha activity (0-1). Inverse Hill function of pO2.');

% Dimensionless ratio avoids SimBiology dimensional analysis error with ^n_HIF
addparameter(model,'pO2_ratio',0,'ValueUnits','dimensionless','ConstantValue',false, ...
    'Notes', 'pO2/pO2_50 ratio (dimensionless). Intermediate for HIF Hill function.');

%% Add Rules for pO2 and HIF
% pO2 = pO2_normal * K / (K + k_O2_demand * cell_demand * vascular_quality_discount)
% Supply: proportional to K (vasculature)
% Vascular quality discount: (1 + VEGF/VEGF_norm_50) — high VEGF makes vessels aberrant
% Demand: total metabolically active cells (C_total + T_total + M_total + aPSC_total + MDSC)
addrule(model, ['pO2 = pO2_normal * V_T.K / (V_T.K + k_O2_demand * ' ...
    '(C_total + T_total + M_total + aPSC_total + V_T.MDSC) * ' ...
    '(1 + V_T.VEGF / VEGF_norm_50))'], 'repeatedAssignment');

addrule(model, 'pO2_ratio = pO2 / pO2_50', 'repeatedAssignment');

% HIF = inverse Hill function of pO2
% At pO2=2.7, pO2_50=7, n_HIF=2: HIF ≈ 0.87 (strongly hypoxic)
addrule(model, 'HIF = 1 / (1 + pO2_ratio^n_HIF)', 'repeatedAssignment');

%% Effect 1: HIF-driven VEGF amplification (positive feedback loop)
% New parallel reaction additive to existing k_vas_Csec*C_total
% Creates: low pO2 -> HIF -> more VEGF -> aberrant vasculature -> lower pO2
reaction = addreaction(model,'null -> V_T.VEGF');
    set(reaction,'ReactionRate','k_VEGF_HIF*C_total*HIF');
    set(reaction,'Notes','HIF-driven VEGF secretion by cancer cells (hypoxia positive feedback)');

%% Effect 2: HIF-driven PD-L1 upregulation (IFNg-independent, constitutive)
% Applied to cancer cell synapses only (not APC — APCs reside in less hypoxic regions)
% Separate ceiling r_PDL1_HIF=3 (vs IFNg ceiling r_PDL1_IFNg=6)
for i = 1:length(model.Compartment)
    comp = model.Compartment(i);
    % Match synapse compartments between CD8 and cancer cells (syn_CD8_C*)
    if ~isempty(regexp(comp.Name, '^syn_CD8_C\d+$', 'once'))
        % Extract cancer cell name from synapse name (e.g., 'C1' from 'syn_CD8_C1')
        Cname = regexprep(comp.Name, '^syn_CD8_', '');
        reaction = addreaction(model, ['null -> ' comp.Name '.PDL1']);
            set(reaction, 'ReactionRate', ['k_PDL1_HIF*HIF*(1 - ' ...
                comp.Name '_PDL1_total/(' Cname '_PDL1_base*r_PDL1_HIF/A_cell))']);
            set(reaction, 'Notes', ['HIF-driven PD-L1 upregulation on ' Cname ' (constitutive, IFNg-independent)']);
    end
end

%% Effect 3: HIF-driven M2 macrophage polarization
% The base M1->M2 rate in macrophage_module.m uses averaged-Hill logic:
%   k_M2_pol*Mac_M1*(H_TGFb + H_IL10 + H_IL6_M2)/3
% Hypoxia adds HIF as a 4th averaged driver (HIF is dimensionless, saturates
% to 1 under hypoxia) so that:
%   k_M2_pol*Mac_M1*(H_TGFb + H_IL10 + H_IL6_M2 + HIF)/4
% i.e. hypoxia contributes a 1/4 fractional lever in parallel with the
% three cytokine axes. Preserves monotonicity and 0..1 range.

for i = 1:length(model.reaction)
    if strcmp(model.reaction(i).reaction, 'V_T.Mac_M1 -> V_T.Mac_M2')
        oldRate = model.reaction(i).ReactionRate;
        % Base rate ends with: ...+ H_IL6_M2)/3
        % Insert HIF as the 4th averaged term and divide by 4.
        newRate = strrep(oldRate, '+ H_IL6_M2)/3', '+ H_IL6_M2 + HIF)/4');
        if ~strcmp(oldRate, newRate)
            model.reaction(i).ReactionRate = newRate;
        end
    end
end

end