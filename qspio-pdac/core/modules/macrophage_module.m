% Macrophage Module
%
% Models macrophage recruitment and polarization in tumor compartment
%
% Inputs: model        -- SimBiology model object with four compartments
%         params       -- object containing model parameter Values, Units, and Notes:
%
% Outputs: model -- SimBiology model object with new macrophage module


function model = macrophage_module(model,params,cancer_types,varargin)

% Optional Inputs
in = inputParser;
addParameter(in,'aCD47',1);
% Parse Inputs
parse(in,varargin{:});
antiCD47 = in.Results.aCD47;

% Add Species
Mac_M1  = addspecies(model.Compartment(3),'Mac_M1',0,'InitialAmountUnits','cell');
    set(Mac_M1,'Notes',['Number of M1 macrophage in the tumor compartment']);
Mac_M2  = addspecies(model.Compartment(3),'Mac_M2',0,'InitialAmountUnits','cell');
    set(Mac_M2,'Notes',['Number of M2 macrophage in the tumor compartment']);
IL12  = addspecies(model.Compartment(3),'IL12',0,'InitialAmountUnits','nanomolarity');
    set(IL12,'Notes',['Concentration of IL-12 in tumor']);
IL10  = addspecies(model.Compartment(3),'IL10',0,'InitialAmountUnits','nanomolarity');
    set(IL10,'Notes',['Concentration of IL-10 in tumor']);

% Add Parameters
k_Mac_rec = addparameter(model,'k_Mac_rec',params.k_Mac_rec.Value,'ValueUnits',params.k_Mac_rec.Units, 'Notes', 'Recruitment rate of macrophage into tumor compartment ');
k_Mac_rec.UserData.Source = params.k_Mac_rec.Notes;
    k_Mac_rec.UserData.derivedFrom = params.k_Mac_rec.derivedFrom;
k_Mac_death = addparameter(model,'k_Mac_death',params.k_Mac_death.Value,'ValueUnits',params.k_Mac_death.Units, 'Notes', 'Death rate of macrophages into tumor compartment ');
k_Mac_death.UserData.Source = params.k_Mac_death.Notes;
    k_Mac_death.UserData.derivedFrom = params.k_Mac_death.derivedFrom;
k_TGFb_Msec = addparameter(model,'k_TGFb_Msec',params.k_TGFb_Msec.Value,'ValueUnits',params.k_TGFb_Msec.Units, 'Notes', 'Secretion rate of TGFb by macrophage in tumor compartment ');
k_TGFb_Msec.UserData.Source = params.k_TGFb_Msec.Notes;
    k_TGFb_Msec.UserData.derivedFrom = params.k_TGFb_Msec.derivedFrom;
k_vas_Msec = addparameter(model,'k_vas_Msec',params.k_vas_Msec.Value,'ValueUnits',params.k_vas_Msec.Units, 'Notes', 'Secretion rate of angiogenic factor by macrophage in tumor compartment ');
k_vas_Msec.UserData.Source = params.k_vas_Msec.Notes;
    k_vas_Msec.UserData.derivedFrom = params.k_vas_Msec.derivedFrom;
k_IL12_sec = addparameter(model,'k_IL12_sec',params.k_IL12_sec.Value,'ValueUnits',params.k_IL12_sec.Units, 'Notes', 'Secretion rate of IL-12 by mAPC in tumor compartment ');
k_IL12_sec.UserData.Source = params.k_IL12_sec.Notes;
    k_IL12_sec.UserData.derivedFrom = params.k_IL12_sec.derivedFrom;
k_IL12_Msec = addparameter(model,'k_IL12_Msec',params.k_IL12_Msec.Value,'ValueUnits',params.k_IL12_Msec.Units, 'Notes', 'Secretion rate of IL-12 by macrophage in tumor compartment ');
k_IL12_Msec.UserData.Source = params.k_IL12_Msec.Notes;
    k_IL12_Msec.UserData.derivedFrom = params.k_IL12_Msec.derivedFrom;
k_IL12_deg = addparameter(model,'k_IL12_deg',params.k_IL12_deg.Value,'ValueUnits',params.k_IL12_deg.Units, 'Notes', 'Degradation rate of IL-12 in tumor compartment ');
k_IL12_deg.UserData.Source = params.k_IL12_deg.Notes;
    k_IL12_deg.UserData.derivedFrom = params.k_IL12_deg.derivedFrom;
k_IL10_sec = addparameter(model,'k_IL10_sec',params.k_IL10_sec.Value,'ValueUnits',params.k_IL10_sec.Units, 'Notes', 'Secretion rate of IL-10 by macrophage in tumor compartment ');
k_IL10_sec.UserData.Source = params.k_IL10_sec.Notes;
    k_IL10_sec.UserData.derivedFrom = params.k_IL10_sec.derivedFrom;
k_IL10_deg = addparameter(model,'k_IL10_deg',params.k_IL10_deg.Value,'ValueUnits',params.k_IL10_deg.Units, 'Notes', 'Degradation rate of IL-10 in tumor compartment ');
k_IL10_deg.UserData.Source = params.k_IL10_deg.Notes;
    k_IL10_deg.UserData.derivedFrom = params.k_IL10_deg.derivedFrom;

k_M2_pol = addparameter(model,'k_M2_pol',params.k_M2_pol.Value,'ValueUnits',params.k_M2_pol.Units, 'Notes', 'Rate of M1 to M2 macrophage polarization ');
k_M2_pol.UserData.Source = params.k_M2_pol.Notes;
    k_M2_pol.UserData.derivedFrom = params.k_M2_pol.derivedFrom;
k_M1_pol = addparameter(model,'k_M1_pol',params.k_M1_pol.Value,'ValueUnits',params.k_M1_pol.Units, 'Notes', 'Rate of M2 to M1 macrophage polarization ');
k_M1_pol.UserData.Source = params.k_M1_pol.Notes;
    k_M1_pol.UserData.derivedFrom = params.k_M1_pol.derivedFrom;
IL10_50 = addparameter(model,'IL10_50',params.IL10_50.Value,'ValueUnits',params.IL10_50.Units, 'Notes', 'Half-maximal IL10 level for M1 to M2 polarization / maintaining Treg function / mAPC inhibition (*STAT6 related) ');
IL10_50.UserData.Source = params.IL10_50.Notes;
    IL10_50.UserData.derivedFrom = params.IL10_50.derivedFrom;
IL12_50 = addparameter(model,'IL12_50',params.IL12_50.Value,'ValueUnits',params.IL12_50.Units, 'Notes', 'Half-maximal IL-12 level for M2 to M1 macrophage polarization ');
IL12_50.UserData.Source = params.IL12_50.Notes;
    IL12_50.UserData.derivedFrom = params.IL12_50.derivedFrom;
k_IFNg_response = addparameter(model,'k_IFNg_response',params.k_IFNg_response.Value,'ValueUnits',params.k_IFNg_response.Units, 'Notes', 'Linear slope of IFNg-driven M2 to M1 polarization in the sub-saturating regime (replaces IFNg_50 Hill)');
k_IFNg_response.UserData.Source = params.k_IFNg_response.Notes;
    k_IFNg_response.UserData.derivedFrom = params.k_IFNg_response.derivedFrom;

k_M1_phago = addparameter(model,'k_M1_phago',params.k_M1_phago.Value,'ValueUnits',params.k_M1_phago.Units, 'Notes', 'Rate of M1 macrophage-mediated killing of cancer cell ');
k_M1_phago.UserData.Source = params.k_M1_phago.Notes;
    k_M1_phago.UserData.derivedFrom = params.k_M1_phago.derivedFrom;

vol_Mcell = addparameter(model,'vol_Mcell',params.vol_Mcell.Value,'ValueUnits',params.vol_Mcell.Units, 'Notes', 'Volume of a single macrophage cell ');
vol_Mcell.UserData.Source = params.vol_Mcell.Notes;
    vol_Mcell.UserData.derivedFrom = params.vol_Mcell.derivedFrom;

first_call = true;
try
    % CCL2 in tumor
    CCL2 = addspecies(model.Compartment(3),'CCL2',0,'InitialAmountUnits','nanomolarity', 'Notes', 'Concentration of CCL2 in the tumor compartment');
    % CCL2 Parameters
    p = addparameter(model,'k_CCL2_sec',params.k_CCL2_sec.Value,'ValueUnits',params.k_CCL2_sec.Units, 'Notes', 'rate of CCL2 secretion ');
    p.UserData.Source = params.k_CCL2_sec.Notes;
        p.UserData.derivedFrom = params.k_CCL2_sec.derivedFrom;
    p = addparameter(model,'k_CCL2_deg',params.k_CCL2_deg.Value,'ValueUnits',params.k_CCL2_deg.Units, 'Notes', 'rate of CCL2 degradation ');
    p.UserData.Source = params.k_CCL2_deg.Notes;
        p.UserData.derivedFrom = params.k_CCL2_deg.derivedFrom;
    p = addparameter(model,'CCL2_50',params.CCL2_50.Value,'ValueUnits',params.CCL2_50.Units, 'Notes', 'Half-maximal CCL2 level of MDSC recruitment ');
    p.UserData.Source = params.CCL2_50.Notes;
        p.UserData.derivedFrom = params.CCL2_50.derivedFrom;
    % CCL2 secretion by MDSCs and Cancer cells
    reaction = addreaction(model,'null -> V_T.CCL2');
        set(reaction,'ReactionRate','k_CCL2_sec*C_total');
        set(reaction,'Notes','CCL2 secretion by MDSCs and cancer cells');
    % CCL2 Degradation
    reaction = addreaction(model,'V_T.CCL2 -> null');
        set(reaction,'ReactionRate','k_CCL2_deg*V_T.CCL2');
        set(reaction,'Notes','CCL2 degradation');
catch
    first_call = false;
end

if ~isempty(sbioselect(model, 'Name', 'VEGF'))
    % Secretion of angiogenic factor
    reaction = addreaction(model,'null -> V_T.VEGF');
        set(reaction,'ReactionRate','k_vas_Msec*V_T.Mac_M2');
        set(reaction,'Notes','Secretion of angiogenic factor by macrophage in tumor');
end

% Add Reactions
% Recruitment of M1 macrophage
reaction = addreaction(model,'null -> V_T.Mac_M1');
    set(reaction,'ReactionRate','k_Mac_rec*V_T*(V_T.CCL2/(V_T.CCL2 + CCL2_50))');
    set(reaction,'Notes','Recruitment of pro-inflammatory M1 macrophage to tumor');
% Macrophage death upon tumor eradication
reaction = addreaction(model,'V_T.Mac_M1 -> null');
    set(reaction,'ReactionRate','k_cell_clear*V_T.Mac_M1*(K_C2_recruit/(C_total^2 + K_C2_recruit))');
    set(reaction,'Notes','Macrophage death upon tumor eradication');
reaction = addreaction(model,'V_T.Mac_M2 -> null');
    set(reaction,'ReactionRate','k_cell_clear*V_T.Mac_M2*(K_C2_recruit/(C_total^2 + K_C2_recruit))');
    set(reaction,'Notes','Macrophage death upon tumor eradication');

% Death of macrophage
reaction = addreaction(model,'V_T.Mac_M1 -> null');
    set(reaction,'ReactionRate','k_Mac_death*V_T.Mac_M1');
    set(reaction,'Notes','Death of macrophage in tumor');
reaction = addreaction(model,'V_T.Mac_M2 -> null');
    set(reaction,'ReactionRate','k_Mac_death*V_T.Mac_M2');
    set(reaction,'Notes','Death of macrophage in tumor');

% Secretion of IL-12 by mAPC in the tumor
reaction = addreaction(model,'null -> V_T.IL12');
    set(reaction,'ReactionRate','k_IL12_sec*V_T.mcDC1');
    set(reaction,'Notes','Secretion of IL-12 by mature cDC1 in the tumor');
% Secretion of IL-12 by M1 macrophage
reaction = addreaction(model,'null -> V_T.IL12');
    set(reaction,'ReactionRate','k_IL12_Msec*V_T.Mac_M1');
    set(reaction,'Notes','Secretion of IL-12 by macrophage in tumor');
% Degradation of IL-12
reaction = addreaction(model,'V_T.IL12 -> null');
    set(reaction,'ReactionRate','k_IL12_deg*V_T.IL12');
    set(reaction,'Notes','Degradation of IL-12 in tumor');
% Secretion of TGFb by M2 macrophages (bioactive fraction of total Matsuo
% 2000 rate; see cytokine_tcell_module notes on f_active_TGFb semantics).
reaction = addreaction(model,'null -> V_T.TGFb');
    set(reaction,'ReactionRate','f_active_TGFb*k_TGFb_Msec*V_T.Mac_M2');
    set(reaction,'Notes','Bioactive TGFb secretion by M2 macrophages (scaled by f_active_TGFb).');

% Secretion of IL-10
reaction = addreaction(model,'null -> V_T.IL10');
    set(reaction,'ReactionRate','k_IL10_sec*V_T.Mac_M2');
    set(reaction,'Notes','Secretion of IL-10 by macrophage in tumor');
% Degradation of IL-10
reaction = addreaction(model,'V_T.IL10 -> null');
    set(reaction,'ReactionRate','k_IL10_deg*V_T.IL10');
    set(reaction,'Notes','Degradation of IL-10 in tumor');

% M1 to M2 polarization (driven by TGF-β, IL-10, and IL-6)
% Averaged Hill logic: each cytokine contributes a weighted share of the
% polarization rate. Replaces the earlier OR-gate
%   1 - (1-H_TGFb)(1-H_IL10)(1-H_IL6_M2)
% which numerically saturates to 1.0 as soon as any two of the three Hills
% sit near their EC50 plateau — producing a "constitutively-maximal"
% polarization rate in PDAC where TGFb, IL-10, and IL-6 are all chronically
% above their bioactive EC50s. Averaging preserves the monotone structure
% (H -> 1 on saturation, H -> 0 when all axes absent), lets each axis
% contribute a 1/3 fractional lever, and matches the multi-subtype biology
% where IL-10, TGFb, and IL-6 drive distinct M2 polarization programs that
% sum rather than short-circuit.
reaction = addreaction(model,'V_T.Mac_M1 -> V_T.Mac_M2');
    set(reaction,'ReactionRate','k_M2_pol*V_T.Mac_M1*(V_T.TGFb/(V_T.TGFb+TGFb_50) + V_T.IL10/(V_T.IL10+IL10_50) + H_IL6_M2)/3');
    set(reaction,'Notes','M1 to M2 macrophage polarization (averaged Hill logic over TGFb, IL-10, IL-6 — max rate at k_M2_pol when all three are saturated).');
% M2 to M1 polarization (driven by IL-12 and IFN-γ)
% Uses "OR" logic: either cytokine can drive polarization, combined effect ≤ 1
% IL-12 is modeled as a saturating Hill (IL12_50 is in the tissue operating range).
% IFN-γ is modeled as a LINEAR term k_IFNg_response * IFNg. Rationale: individual
% macrophages respond to IFNg with a ~1-nM single-cell Hill EC50, but IFNg signaling
% is spatially confined to ~30-40 um around producing TILs (Centofanti 2023 PMID
% 37603742). Since V_T.IFNg and V_T.Mac_M2 are both well-mixed compartments here, the
% population-level polarization drive tracks the fraction of macrophages within a TIL
% signaling zone, which scales linearly with bulk IFNg (not with the single-cell Hill).
% k_IFNg_response is therefore a spatial-coverage slope, not a receptor EC50.
% max(0, ...) clamps the (1 - linear) factor at 0 to avoid negative contributions at
% pathological high bulk IFNg (reaction saturates at k_M1_pol, matching the old Hill OR).
reaction = addreaction(model,'V_T.Mac_M2 -> V_T.Mac_M1');
    set(reaction,'ReactionRate','k_M1_pol*V_T.Mac_M2*(1 - (1-V_T.IL12/(V_T.IL12+IL12_50))*max(0, 1 - k_IFNg_response*V_T.IFNg))');
    set(reaction,'Notes','M2 to M1 macrophage polarization in tumor (IL-12 Hill + IFNg linear sub-saturating - OR logic, max rate capped at k_M1_pol)');

model = phagocytosis_module(model,params,'aCD47',antiCD47);

IL10_50_phago = addparameter(model,'IL10_50_phago',params.IL10_50_phago.Value,'ValueUnits',params.IL10_50_phago.Units, 'Notes', 'Half-maximal IL-10 level for phagocytosis by macrophage ');
IL10_50_phago.UserData.Source = params.IL10_50_phago.Notes;
    IL10_50_phago.UserData.derivedFrom = params.IL10_50_phago.derivedFrom;
K_M1_phago = addparameter(model,'K_M1_phago',params.K_M1_phago.Value,'ValueUnits',params.K_M1_phago.Units, 'Notes', 'Association coefficient for macrophage and cancer cell ');
K_M1_phago.UserData.Source = params.K_M1_phago.Notes;
    K_M1_phago.UserData.derivedFrom = params.K_M1_phago.derivedFrom;

M_total = addparameter(model,'M_total',0,'ValueUnits','cell','ConstantValue',false, 'Notes', 'Total number of macrophages ');
H_IL10 = addparameter(model,'H_IL10',1,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function for IL10-mediated M2-like macrophage polarization / Treg functionality / inhibition on Ag presentaion ');
H_IL10_phago = addparameter(model,'H_IL10_phago',1,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function for IL10-mediated inhibition on phagocytosis ');
H_IL12 = addparameter(model,'H_IL12',1,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function for IL12 ');

% IL-6-mediated enhancement of M2 polarization (PDAC-specific)
% Create Hill function (will be 0 unless rule is added by fibroblast module)
H_IL6_M2 = addparameter(model,'H_IL6_M2',0,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function for IL-6 promotion of M2 macrophage polarization');
% Rule will be added by fibroblast_module_enhanced if IL-6 exists

addrule(model,'M_total = V_T.Mac_M1+V_T.Mac_M2','repeatedAssignment');
addrule(model,'H_IL10 = V_T.IL10/(IL10_50+V_T.IL10)','repeatedAssignment');
addrule(model,'H_IL10_phago = V_T.IL10/(IL10_50_phago+V_T.IL10)','repeatedAssignment');
addrule(model,'H_IL12 = V_T.IL12/(IL12_50+V_T.IL12)','repeatedAssignment');

% Get Model Rules for Updating
model_rules = get(model,'Rules');

% Update tumor Volume (Rule 1)
volume_rule = model_rules(1);
rule = get(volume_rule,'Rule');
set(volume_rule,'Rule',[rule '+M_total*vol_Mcell/Ve_T']); % 4990 µm^3/cell

% Update Cancer Killing by Macrophage
if (exist('cancer_types','var'))
    for i = 1:length(cancer_types)
        reaction = addreaction(model, ['V_T.' cancer_types{i} ' -> V_T.C_x']);
            % set(reaction,'ReactionRate',['k_M1_phago*V_T.' cancer_types{i} '*V_T.Mac_M1/(V_T.Mac_M1+K_M1_phago*V_T.' cancer_types{i} '+cell)*(1-H_Mac_C)*(1-H_IL10_phago)']);
            set(reaction,'ReactionRate',['k_M1_phago*V_T.' cancer_types{i} '*V_T.Mac_M1/(V_T.Mac_M1+K_M1_phago*C_total+cell)*(1-H_Mac_C)*(1-H_IL10_phago)']);
            set(reaction,'Notes','M1 macrophage-mediated killing of cancer cell');
        % Update antigen release rate (Rule 5)
        rule = get(model_rules(5),'Rule');
           % set(model_rules(5),'Rule',[rule,'+k_M1_phago*V_T.' cancer_types{i} '*V_T.Mac_M1/(V_T.Mac_M1+K_M1_phago*V_T.' cancer_types{i} '+cell)*(1-H_Mac_C)*(1-H_IL10_phago)']);
            set(model_rules(5),'Rule',[rule,'+k_M1_phago*V_T.' cancer_types{i} '*V_T.Mac_M1/(V_T.Mac_M1+K_M1_phago*C_total+cell)*(1-H_Mac_C)*(1-H_IL10_phago)']);
    end
end

% Update Treg-mediated Teff exhaustion (IL-10 amplifies Treg suppression)
for i = 1:length(model.reaction)
    if strcmp(model.reaction(i).reaction, 'V_T.CD8 -> V_T.CD8_exh') && ~isempty(strfind(model.reaction(i).ReactionRate, 'k_Treg_CD8_exh'))
        model.reaction(i).ReactionRate = [model.reaction(i).ReactionRate, '*H_IL10'];
    end
end
