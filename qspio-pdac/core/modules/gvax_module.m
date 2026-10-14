function model = gvax_module(model, params, varargin)
% GVAX module with intradermal injection site compartment
% - GVAX cells administered to intradermal injection site (V_ID)
% - Local APC recruitment and maturation at injection site
% - Mature APCs migrate from injection site to lymph nodes
% - Optional: cyclophosphamide for Treg depletion
%
% Dosing is handled through schedule_dosing('GVAX') which doses into V_ID.GVAX_cells
%
% Optional parameters:
%   'add_cyclophosphamide' - logical flag to add Cy for Treg depletion (default: false)

% Parse optional inputs
p = inputParser;
addParameter(p, 'add_cyclophosphamide', false, @islogical);
parse(p, varargin{:});
add_cy = p.Results.add_cyclophosphamide;

%% ---- Add intradermal injection site compartment and species ----
% Create intradermal injection site compartment
V_ID = sbioselect(model, 'Name', 'V_ID', 'Type', 'compartment');
if isempty(V_ID)
    V_ID = addcompartment(model, 'V_ID', 'CapacityUnits', 'liter');
    set(V_ID, 'Notes', 'Intradermal injection site compartment');
end

% Add GVAX_cells species in injection site (target for dosing)
gvax_cells = sbioselect(model, 'Name', 'GVAX_cells', 'Type', 'species');
if isempty(gvax_cells)
    gvax_cells = addspecies(V_ID, 'GVAX_cells', 0, 'InitialAmountUnits', 'cell');
    set(gvax_cells, 'Notes', 'GVAX vaccine cells at intradermal injection site');
end

% Add APC species at injection site
apc_id = sbioselect(model, 'Name', 'APC', 'Type', 'species', 'Parent', V_ID);
if isempty(apc_id)
    apc_id = addspecies(V_ID, 'APC', 0, 'InitialAmountUnits', 'cell');
    set(apc_id, 'Notes', 'Naive APCs recruited to injection site');
end

% Add mature APC species at injection site
mapc_id = sbioselect(model, 'Name', 'mAPC', 'Type', 'species', 'Parent', V_ID);
if isempty(mapc_id)
    mapc_id = addspecies(V_ID, 'mAPC', 0, 'InitialAmountUnits', 'cell');
    set(mapc_id, 'Notes', 'Mature APCs with antigen at injection site');
end

% Add GM-CSF species at injection site
gmcsf_id = sbioselect(model, 'Name', 'GMCSF', 'Type', 'species', 'Parent', V_ID);
if isempty(gmcsf_id)
    gmcsf_id = addspecies(V_ID, 'GMCSF', 0, 'InitialAmountUnits', 'nanomole');
    set(gmcsf_id, 'Notes', 'GM-CSF amount at injection site');
end

% Add P1_GVAX antigen species at injection site
p1_gvax_id = sbioselect(model, 'Name', 'P1_GVAX', 'Type', 'species', 'Parent', V_ID);
if isempty(p1_gvax_id)
    p1_gvax_id = addspecies(V_ID, 'P1_GVAX', 0, 'InitialAmountUnits', 'nanomole');
    set(p1_gvax_id, 'Notes', 'GVAX-derived antigen at injection site');
end

% Load parameters from params structure

% GVAX cell clearance at injection site
k_GVAX = sbioselect(model,'Type','parameter','Name','k_GVAX_clearance');
if isempty(k_GVAX), k_GVAX = addparameter(model,'k_GVAX_clearance',params.k_GVAX_clearance.Value,'ValueUnits',params.k_GVAX_clearance.Units); end
set(k_GVAX, 'Notes', params.k_GVAX_clearance.Notes);

% Injection site volume
V_ID_vol = sbioselect(model,'Type','parameter','Name','V_ID_volume');
if isempty(V_ID_vol), V_ID_vol = addparameter(model,'V_ID_volume',params.V_ID_volume.Value,'ValueUnits',params.V_ID_volume.Units); end
set(V_ID_vol, 'Notes', params.V_ID_volume.Notes);
set(V_ID, 'Capacity', V_ID_vol.Value);

% GM-CSF production and degradation parameters
k_GMCSF_prod = sbioselect(model,'Type','parameter','Name','k_GMCSF_prod');
if isempty(k_GMCSF_prod), k_GMCSF_prod = addparameter(model,'k_GMCSF_prod',params.k_GMCSF_prod.Value,'ValueUnits',params.k_GMCSF_prod.Units); end
set(k_GMCSF_prod, 'Notes', params.k_GMCSF_prod.Notes);

k_GMCSF_deg = sbioselect(model,'Type','parameter','Name','k_GMCSF_deg');
if isempty(k_GMCSF_deg), k_GMCSF_deg = addparameter(model,'k_GMCSF_deg',params.k_GMCSF_deg.Value,'ValueUnits',params.k_GMCSF_deg.Units); end
set(k_GMCSF_deg, 'Notes', params.k_GMCSF_deg.Notes);

% APC recruitment to injection site
k_APC_recruit_ID = sbioselect(model,'Type','parameter','Name','k_APC_recruit_ID');
if isempty(k_APC_recruit_ID), k_APC_recruit_ID = addparameter(model,'k_APC_recruit_ID',params.k_APC_recruit_ID.Value,'ValueUnits',params.k_APC_recruit_ID.Units); end
set(k_APC_recruit_ID, 'Notes', params.k_APC_recruit_ID.Notes);

% GM-CSF effect parameters for APC recruitment (GM-CSF's correct role; maturation
% is HMGB1/DAMP-licensed via H_HMGB1_ID, not GM-CSF)
EC50_GMCSF = sbioselect(model,'Type','parameter','Name','EC50_GMCSF');
if isempty(EC50_GMCSF), EC50_GMCSF = addparameter(model,'EC50_GMCSF',params.EC50_GMCSF.Value,'ValueUnits',params.EC50_GMCSF.Units); end
set(EC50_GMCSF, 'Notes', params.EC50_GMCSF.Notes);

k_APC_mature_ID = sbioselect(model,'Type','parameter','Name','k_APC_mature_ID');
if isempty(k_APC_mature_ID), k_APC_mature_ID = addparameter(model,'k_APC_mature_ID',params.k_APC_mature_ID.Value,'ValueUnits',params.k_APC_mature_ID.Units); end
set(k_APC_mature_ID, 'Notes', params.k_APC_mature_ID.Notes);

% APC migration from injection site to lymph nodes
k_APC_mig_ID_LN = sbioselect(model,'Type','parameter','Name','k_APC_mig_ID_LN');
if isempty(k_APC_mig_ID_LN), k_APC_mig_ID_LN = addparameter(model,'k_APC_mig_ID_LN',params.k_APC_mig_ID_LN.Value,'ValueUnits',params.k_APC_mig_ID_LN.Units); end
set(k_APC_mig_ID_LN, 'Notes', params.k_APC_mig_ID_LN.Notes);

% APC death at injection site
k_APC_death_ID = sbioselect(model,'Type','parameter','Name','k_APC_death_ID');
if isempty(k_APC_death_ID), k_APC_death_ID = addparameter(model,'k_APC_death_ID',params.k_APC_death_ID.Value,'ValueUnits',params.k_APC_death_ID.Units); end
set(k_APC_death_ID, 'Notes', params.k_APC_death_ID.Notes);

% Antigen consumption per APC maturation
sigma_P1_per_mAPC = sbioselect(model,'Type','parameter','Name','sigma_P1_per_mAPC');
if isempty(sigma_P1_per_mAPC), sigma_P1_per_mAPC = addparameter(model,'sigma_P1_per_mAPC',params.sigma_P1_per_mAPC.Value,'ValueUnits',params.sigma_P1_per_mAPC.Units); end
set(sigma_P1_per_mAPC, 'Notes', params.sigma_P1_per_mAPC.Notes);

% EC50 for antigen-dependent APC maturation
EC50_P1_mature = sbioselect(model,'Type','parameter','Name','EC50_P1_mature');
if isempty(EC50_P1_mature), EC50_P1_mature = addparameter(model,'EC50_P1_mature',params.EC50_P1_mature.Value,'ValueUnits',params.EC50_P1_mature.Units); end
set(EC50_P1_mature, 'Notes', params.EC50_P1_mature.Notes);

% Capacity and baseline recruitment
K_APC_ID = sbioselect(model,'Type','parameter','Name','K_APC_ID');
if isempty(K_APC_ID), K_APC_ID = addparameter(model,'K_APC_ID',params.K_APC_ID.Value,'ValueUnits','cell'); end

k_APC_recruit0 = sbioselect(model,'Type','parameter','Name','k_APC_recruit0');
if isempty(k_APC_recruit0), k_APC_recruit0 = addparameter(model,'k_APC_recruit0',params.k_APC_recruit0.Value,'ValueUnits',params.k_APC_recruit0.Units); end

% Separate mature-DC death
k_mAPC_death_ID = sbioselect(model,'Type','parameter','Name','k_mAPC_death_ID');
if isempty(k_mAPC_death_ID), k_mAPC_death_ID = addparameter(model,'k_mAPC_death_ID',params.k_mAPC_death_ID.Value,'ValueUnits',params.k_mAPC_death_ID.Units); end

% Antigen yield per cell death and degradation/drain
y_Ag_per_cell = sbioselect(model,'Type','parameter','Name','y_Ag_per_cell');
if isempty(y_Ag_per_cell), y_Ag_per_cell = addparameter(model,'y_Ag_per_cell',params.y_Ag_per_cell.Value,'ValueUnits','nanomole/cell'); end

k_P1_deg_ID = sbioselect(model,'Type','parameter','Name','k_P1_deg_ID');
if isempty(k_P1_deg_ID), k_P1_deg_ID = addparameter(model,'k_P1_deg_ID',params.k_P1_deg_ID.Value,'ValueUnits',params.k_P1_deg_ID.Units); end

% APC migration saturation
K_APC_mig = sbioselect(model,'Type','parameter','Name','K_APC_mig');
if isempty(K_APC_mig), K_APC_mig = addparameter(model,'K_APC_mig',params.K_APC_mig.Value,'ValueUnits',params.K_APC_mig.Units); end

% Antigen-transfer rate: migratory mcDC2 -> resident cDC1 cross-licensing in LN
k_cDC1_Ag_transfer = sbioselect(model,'Type','parameter','Name','k_cDC1_Ag_transfer');
if isempty(k_cDC1_Ag_transfer), k_cDC1_Ag_transfer = addparameter(model,'k_cDC1_Ag_transfer',params.k_cDC1_Ag_transfer.Value,'ValueUnits',params.k_cDC1_Ag_transfer.Units); end
set(k_cDC1_Ag_transfer, 'Notes', params.k_cDC1_Ag_transfer.Notes);

% (f_mAPC_mcDC1_fate removed in mid-tier #15: the direct depot-moDC->mcDC1
% cross-priming route was retired in favor of the host-cDC1 antigen-handoff
% route -- see the migration block below.)

% Vaccine-antigen presentation rate: feeds GVAX antigen into the shared
% endosomal antigen-1 pool (V_e.P1) so the vaccine raises H_P1 (pMHC display),
% not just the DC count. Parallel to tumor-antigen uptake k_up (same 1/day/cell).
k_GVAX_present = sbioselect(model,'Type','parameter','Name','k_GVAX_present');
if isempty(k_GVAX_present), k_GVAX_present = addparameter(model,'k_GVAX_present',params.k_GVAX_present.Value,'ValueUnits',params.k_GVAX_present.Units); end
set(k_GVAX_present, 'Notes', params.k_GVAX_present.Notes);

%% ---- Add Hill function for GM-CSF effects at injection site ----
% Hill function for GM-CSF effects (normalized to [0,1])
H_GMCSF_ID = sbioselect(model,'Type','parameter','Name','H_GMCSF_ID');
if isempty(H_GMCSF_ID), H_GMCSF_ID = addparameter(model,'H_GMCSF_ID',0,'ValueUnits','dimensionless','ConstantValue',false); end

% Get lymph node compartment for APC migration
V_LN = sbioselect(model, 'Name', 'V_LN', 'Type', 'compartment');

%% ---- Add Hill function rule for injection site GM-CSF ----
gmcsfRuleTxt_ID = 'H_GMCSF_ID = (V_ID.GMCSF / V_ID) / (EC50_GMCSF + (V_ID.GMCSF / V_ID))';

% Replace existing rules if present, else add
R = sbioselect(model,'Type','rule');
didReplaceGMCSF_ID = false;
for k = 1:numel(R)
    if contains(strrep(R(k).Rule,' ',''), 'H_GMCSF_ID=')
        set(R(k),'Rule',gmcsfRuleTxt_ID); didReplaceGMCSF_ID = true;
    end
end
if ~didReplaceGMCSF_ID
    addrule(model, gmcsfRuleTxt_ID, 'repeatedAssignment');
end

%% ---- Add HMGB1 (DAMP) licensing of injection-site APC maturation ----
% GM-CSF recruits APCs to the depot (recruitment reaction below) but does NOT
% license their maturation -- a GM-CSF field without danger signals drives
% accumulation of immature/tolerogenic DCs. Maturation is DAMP/CD40L/IFN-licensed.
% The irradiated GVAX cells die (k_GVAX_clearance * V_ID.GVAX_cells, cell/day) and
% passively release HMGB1, the representative maturation DAMP. H_HMGB1_ID is a Hill
% on that depot death FLUX, mirroring the tumor-side
% H_HMGB1 = R_Tcell/(R_Tcell+HMGB1_50) in APC_module.m.
% (Mechanism-audit mid-tier cA-G1 / x1-GVAX-B01: GM-CSF was the wrong maturation
% mediator. Pairs with the CD40L-licensing consensus row.)
HMGB1_50_ID = sbioselect(model,'Type','parameter','Name','HMGB1_50_ID');
if isempty(HMGB1_50_ID), HMGB1_50_ID = addparameter(model,'HMGB1_50_ID',params.HMGB1_50_ID.Value,'ValueUnits',params.HMGB1_50_ID.Units); end
set(HMGB1_50_ID, 'Notes', params.HMGB1_50_ID.Notes);

H_HMGB1_ID = sbioselect(model,'Type','parameter','Name','H_HMGB1_ID');
if isempty(H_HMGB1_ID), H_HMGB1_ID = addparameter(model,'H_HMGB1_ID',0,'ValueUnits','dimensionless','ConstantValue',false); end

hmgb1RuleTxt_ID = 'H_HMGB1_ID = (k_GVAX_clearance * V_ID.GVAX_cells) / (k_GVAX_clearance * V_ID.GVAX_cells + HMGB1_50_ID)';
R = sbioselect(model,'Type','rule');
didReplaceHMGB1_ID = false;
for k = 1:numel(R)
    if contains(strrep(R(k).Rule,' ',''), 'H_HMGB1_ID=')
        set(R(k),'Rule',hmgb1RuleTxt_ID); didReplaceHMGB1_ID = true;
    end
end
if ~didReplaceHMGB1_ID
    addrule(model, hmgb1RuleTxt_ID, 'repeatedAssignment');
end

%% ---- Add APC maturation flux parameter and rule ----
% APC maturation flux (used for both maturation and antigen consumption)
J_mature_ID = sbioselect(model,'Type','parameter','Name','J_mature_ID');
if isempty(J_mature_ID), J_mature_ID = addparameter(model,'J_mature_ID',0,'ValueUnits','cell/day','ConstantValue',false); end

% Define maturation flux rule
% Maturation licensed by HMGB1 (DAMP) from dying GVAX cells, gated by vaccine-antigen
% presence -- the textbook two-signal (danger + antigen) picture. GM-CSF stays in
% recruitment only (its correct role), not here.
maturationFluxRuleTxt_ID = 'J_mature_ID = k_APC_mature_ID * H_HMGB1_ID * V_ID.APC * (V_ID.P1_GVAX / V_ID) / ((V_ID.P1_GVAX / V_ID) + EC50_P1_mature)';

% Replace existing rules if present, else add
didReplaceMaturation_ID = false;
for k = 1:numel(R)
    if contains(strrep(R(k).Rule,' ',''), 'J_mature_ID=')
        set(R(k),'Rule',maturationFluxRuleTxt_ID); didReplaceMaturation_ID = true;
    end
end
if ~didReplaceMaturation_ID
    addrule(model, maturationFluxRuleTxt_ID, 'repeatedAssignment');
end

%% ---- Path A: vaccine-driven epitope spreading / clonal recruitment ----
% GVAX (whole-cell, GM-CSF) broadens the tumor-specific CD8 repertoire: new clones
% are cross-primed against antigens not in the original response (Thomas 2004 J Exp
% Med, PMID 15289501 -- mesothelin-specific CD8 induced by a vaccine not defining
% that antigen, via APC cross-priming). The model's n_CD8_clones (tumor-specific
% clone count) is otherwise constant, so the committed-clone base aCD8 is pinned and
% cannot broaden under vaccine (cd8_fc deficit Path A; see
% notes/calibration/cd8_fc_vaccine_expansion_deficit.md). Here we scale the recruited
% clone count in the naive-CD8 -> aCD8 commitment reaction by a vaccine-antigen-gated
% spreading factor (1 + f_epitope_spread*H_P1_spread). H_P1_spread reuses the same
% P1_GVAX-concentration Hill the maturation flux uses, so it is identically 0 at
% diagnosis (P1_GVAX = 0 -> multiplier = 1, baseline untouched) and rises under GVAX
% (~0.83 at d21). Only the commitment reaction (uniquely identified by the
% H_CD4_help*n_CD8_clones substring) is rewritten; the n_CD8_clones PARAMETER is
% unchanged, so the antigen_rate.m and TCR_KPR_module.m clone-count usages are
% unaffected.
% CD4 and CD8 use one effective intensity. The active treatment targets do not
% identify two independent amplitudes, so pathway-specific presentation detail is
% retained in the surrounding reactions rather than asserted as a fitted ratio.
f_spread = sbioselect(model,'Type','parameter','Name','f_epitope_spread');
if isempty(f_spread)
    addparameter(model,'f_epitope_spread',params.f_epitope_spread.Value, ...
        'ValueUnits',params.f_epitope_spread.Units,'Notes',params.f_epitope_spread.Notes);
end
H_P1_spread = sbioselect(model,'Type','parameter','Name','H_P1_spread');
if isempty(H_P1_spread)
    addparameter(model,'H_P1_spread',0,'ValueUnits','dimensionless','ConstantValue',false, ...
        'Notes','Vaccine-antigen Hill gating epitope-spreading clonal recruitment (Path A); 0 at diagnosis');
    addrule(model,'H_P1_spread = (V_ID.P1_GVAX / V_ID) / ((V_ID.P1_GVAX / V_ID) + EC50_P1_mature)','repeatedAssignment');
end
rxnACD8 = sbioselect(model,'Type','reaction','Where','ReactionRate','regexp','H_CD4_help\*n_CD8_clones');
if isempty(rxnACD8)
    warning('gvax_module:noACD8commit', ...
        'Naive-CD8->aCD8 commitment reaction (H_CD4_help*n_CD8_clones) not found; epitope-spreading clonal recruitment not applied.');
else
    for i = 1:numel(rxnACD8)
        rate = get(rxnACD8(i),'ReactionRate');
        newrate = strrep(rate,'n_CD8_clones','(n_CD8_clones*(1 + f_epitope_spread*H_P1_spread))');
        if ~strcmp(rate,newrate)
            set(rxnACD8(i),'ReactionRate',newrate);
        end
    end
end

% Path A (CD4 arm): the helper repertoire broadens under GVAX by the same cross-
% priming mechanism, and is in fact where GM-CSF acts most directly (GM-CSF recruits/
% matures the cDC2/moDCs that prime CD4 -- see V_LN.mcDC2 routing). CD4 help is also
% mechanistically central to epitope spreading: sustained cross-presentation of newly
% released tumor antigen requires CD40L licensing (Thomas 2004, same provenance as the
% CD8 arm). Without this term the CD4 commitment is gated only by mcDC2*H_P1_II*nCD4 --
% and a trace shows the naive-CD4 pool craters under vaccine (nCD4 ~0.27x at d21), so
% the antigen-presentation gain is cancelled and V_LN.aTh barely rises (1.19x) while
% V_LN.aCD8 (which has this term) reaches 3.9x. The clone factor sits on the aTh
% PRODUCTION (null -> V_LN.aTh) not the naive-pool drain, so scaling it boosts helper
% output without accelerating the naive depletion. It uses the same effective
% spreading intensity as CD8. The reaction is uniquely
% identified by the H_P1_II*V_LN.nCD4*n_CD4_clones substring; the n_CD4_clones PARAMETER
% is unchanged so antigen_rate.m / TCR_KPR_module.m are untouched.
rxnATh = sbioselect(model,'Type','reaction','Where','ReactionRate','regexp','H_P1_II\*V_LN\.nCD4\*n_CD4_clones');
if isempty(rxnATh)
    warning('gvax_module:noAThcommit', ...
        'Naive-CD4->aTh commitment reaction (H_P1_II*V_LN.nCD4*n_CD4_clones) not found; CD4 epitope-spreading clonal recruitment not applied.');
else
    for i = 1:numel(rxnATh)
        rate = get(rxnATh(i),'ReactionRate');
        newrate = strrep(rate,'n_CD4_clones','(n_CD4_clones*(1 + f_epitope_spread*H_P1_spread))');
        if ~strcmp(rate,newrate)
            set(rxnATh(i),'ReactionRate',newrate);
        end
    end
end

%% ---- Vaccine-driven intratumoral CD8 proliferation ----
% Retained from the retired TLA/urelumab module in C13. This is a bulk GVAX
% response, not a TLA state: vaccine antigen gates an incremental saturating
% source on the existing tumor CD8 pool. It is identically zero without GVAX.
V_T = sbioselect(model,'Name','V_T','Type','compartment');
k_CD8_T_vac_pro = sbioselect(model,'Type','parameter','Name','k_CD8_T_vac_pro');
if isempty(k_CD8_T_vac_pro)
    k_CD8_T_vac_pro = addparameter(model,'k_CD8_T_vac_pro', ...
        params.k_CD8_T_vac_pro.Value,'ValueUnits',params.k_CD8_T_vac_pro.Units);
end
set(k_CD8_T_vac_pro,'Notes',params.k_CD8_T_vac_pro.Notes);
k_CD8_T_vac_pro.UserData.Source = params.k_CD8_T_vac_pro.Notes;
k_CD8_T_vac_pro.UserData.derivedFrom = params.k_CD8_T_vac_pro.derivedFrom;

vacProRxn = addreaction(model,'null -> V_T.CD8');
set(vacProRxn,'Name','GVAX_CD8_tumor_proliferation');
set(vacProRxn,'ReactionRate', ...
    'k_CD8_T_vac_pro*V_T.CD8*H_P1_spread/(1 + V_T.CD8/(rho_CD8_max*V_T + cell))');
set(vacProRxn,'Notes', ...
    'GVAX-antigen-gated bulk intratumoral CD8 proliferation; retained in C13 without a TLA state model.');

%% ---- Injection site GVAX cell dynamics ----
% GVAX cell clearance at injection site
clearanceRxn = sbioselect(model, 'Type', 'reaction', 'Name', 'GVAX_clearance_ID');
if isempty(clearanceRxn)
    clearanceRxn = addreaction(model, 'V_ID.GVAX_cells -> null');
    set(clearanceRxn, 'Name', 'GVAX_clearance_ID');
    set(clearanceRxn, 'ReactionRate', 'k_GVAX_clearance * V_ID.GVAX_cells');
    set(clearanceRxn, 'Notes', 'GVAX cell clearance at injection site');
end

% GM-CSF production by GVAX cells at injection site
gmcsfProdRxn = sbioselect(model, 'Type', 'reaction', 'Name', 'GMCSF_production_ID');
if isempty(gmcsfProdRxn)
    gmcsfProdRxn = addreaction(model, 'null -> V_ID.GMCSF');
    set(gmcsfProdRxn, 'Name', 'GMCSF_production_ID');
    set(gmcsfProdRxn, 'ReactionRate', 'k_GMCSF_prod * V_ID.GVAX_cells');
    set(gmcsfProdRxn, 'Notes', 'GM-CSF production by GVAX cells at injection site');
end

% GM-CSF degradation at injection site
gmcsfDegRxn = sbioselect(model, 'Type', 'reaction', 'Name', 'GMCSF_degradation_ID');
if isempty(gmcsfDegRxn)
    gmcsfDegRxn = addreaction(model, 'V_ID.GMCSF -> null');
    set(gmcsfDegRxn, 'Name', 'GMCSF_degradation_ID');
    set(gmcsfDegRxn, 'ReactionRate', 'k_GMCSF_deg * V_ID.GMCSF');
    set(gmcsfDegRxn, 'Notes', 'GM-CSF degradation at injection site');
end

% Antigen release by GVAX cells at injection site
antigenReleaseRxn = sbioselect(model,'Type','reaction','Name','GVAX_antigen_release_ID');
if isempty(antigenReleaseRxn)
    antigenReleaseRxn = addreaction(model,'null -> V_ID.P1_GVAX');
    set(antigenReleaseRxn,'Name','GVAX_antigen_release_ID');
end
set(antigenReleaseRxn,'ReactionRate','y_Ag_per_cell * k_GVAX_clearance * V_ID.GVAX_cells');

% Antigen degradation/dispersion
p1DegRxn = sbioselect(model,'Type','reaction','Name','P1_degradation_ID');
if isempty(p1DegRxn)
    p1DegRxn = addreaction(model,'V_ID.P1_GVAX -> null');
    set(p1DegRxn,'Name','P1_degradation_ID');
end
set(p1DegRxn,'ReactionRate','k_P1_deg_ID * V_ID.P1_GVAX');

%% ---- APC dynamics at injection site ----
% APC recruitment to injection site (GM-CSF mediated)
apcRecruitRxn = sbioselect(model,'Type','reaction','Name','APC_recruitment_ID');
if isempty(apcRecruitRxn)
    apcRecruitRxn = addreaction(model,'null -> V_ID.APC');
    set(apcRecruitRxn,'Name','APC_recruitment_ID');
end
set(apcRecruitRxn,'ReactionRate',...
  '(k_APC_recruit0 + k_APC_recruit_ID * H_GMCSF_ID) * K_APC_ID /(K_APC_ID + V_ID.APC + V_ID.mAPC)');

% APC maturation at injection site (antigen + GM-CSF mediated)
apcMatureRxn = sbioselect(model, 'Type', 'reaction', 'Name', 'APC_maturation_ID');
if isempty(apcMatureRxn)
    apcMatureRxn = addreaction(model, 'V_ID.APC -> V_ID.mAPC');
    set(apcMatureRxn, 'Name', 'APC_maturation_ID');
    set(apcMatureRxn, 'ReactionRate', 'J_mature_ID');
    set(apcMatureRxn, 'Notes', 'APC maturation with antigen uptake at injection site');
end

% Antigen consumption tied to APC maturation flux
antigenConsumptionRxn = sbioselect(model, 'Type', 'reaction', 'Name', 'P1_consumption_ID');
if isempty(antigenConsumptionRxn)
    antigenConsumptionRxn = addreaction(model, 'V_ID.P1_GVAX -> null');
    set(antigenConsumptionRxn, 'Name', 'P1_consumption_ID');
    set(antigenConsumptionRxn, 'ReactionRate', 'sigma_P1_per_mAPC * J_mature_ID');
    set(antigenConsumptionRxn, 'Notes', 'Antigen consumption tied to APC maturation events');
end

% Vaccine-antigen presentation: feed GVAX antigen into the shared endosomal
% antigen-1 pool so the vaccine raises the pMHC display signal H_P1, not just the
% mature-DC count. Mechanism-audit consensus #2 (GVAX antigen never feeds H_P1):
% GVAX-matured DCs took up/consumed vaccine antigen (above) but merged into the
% shared V_LN.mcDC1/mcDC2 pools whose priming-per-DC was gated by the TUMOR-derived
% H_P1 only -> the vaccine's dose->presentation coupling was missing. This source
% is a direct copy of the tumor-antigen endosomal uptake in antigen_module.m
% (null -> V_e.P1 at k_up*cell*V_T.P1*V_e), swapping the tumor free-antigen
% concentration V_T.P1 for the depot vaccine-antigen concentration V_ID.P1_GVAX/V_ID
% and k_up for k_GVAX_present. The *V_e factor (matching the tumor form) makes the
% steady-state endosomal concentration V_e-independent and on the same scale as
% tumor presentation; without it the source/(k_P_deg*V_e) ratio blows up as 1/V_e
% and pins H_P1 at 1. k_GVAX_present (1/day/cell, parallel to k_up=14.4) is the
% vaccine presentation-uptake efficiency; SBI-varied, submodel anchor pending.
% No GM-CSF gate: the depot antigen concentration provides the on/off, and the
% GM-CSF adjuvant effect is already in the DC recruitment/maturation (count) steps.
gvaxPresentRxn = sbioselect(model, 'Type', 'reaction', 'Name', 'P1_GVAX_presentation_ID');
if isempty(gvaxPresentRxn)
    gvaxPresentRxn = addreaction(model, 'null -> V_e.P1');
    set(gvaxPresentRxn, 'Name', 'P1_GVAX_presentation_ID');
    set(gvaxPresentRxn, 'ReactionRate', 'k_GVAX_present * cell * (V_ID.P1_GVAX / V_ID) * V_e');
    set(gvaxPresentRxn, 'Notes', 'Vaccine antigen feeds the shared endosomal pMHC pool (raises H_P1); copy of tumor-antigen uptake with k_up->k_GVAX_present, V_T.P1->V_ID.P1_GVAX/V_ID (audit consensus #2)');
end

% APC death at injection site
apcDeathRxn = sbioselect(model, 'Type', 'reaction', 'Name', 'APC_death_ID');
if isempty(apcDeathRxn)
    apcDeathRxn = addreaction(model, 'V_ID.APC -> null');
    set(apcDeathRxn, 'Name', 'APC_death_ID');
    set(apcDeathRxn, 'ReactionRate', 'k_APC_death_ID * V_ID.APC');
    set(apcDeathRxn, 'Notes', 'APC death at injection site');
end

% Mature APC death at injection site
mapcDeathRxn = sbioselect(model,'Type','reaction','Name','mAPC_death_ID');
if isempty(mapcDeathRxn)
    mapcDeathRxn = addreaction(model,'V_ID.mAPC -> null');
    set(mapcDeathRxn,'Name','mAPC_death_ID');
end
set(mapcDeathRxn,'ReactionRate','k_mAPC_death_ID * V_ID.mAPC');

%% ---- APC migration from injection site to lymph nodes ----
if ~isempty(V_LN)
  % All depot mAPC migrate to the LN as mcDC2 (MHC-II priming + antigen carrier).
  % CD8 cross-priming flows through the antigen-handoff route below, NOT through a
  % direct depot-moDC->mcDC1 path (mid-tier #15).
  %
  % The former direct route (V_ID.mAPC -> V_LN.mcDC1 at f_mAPC_mcDC1_fate) was
  % removed. moDCs CAN cross-prime CD8 in principle (Briseno 2016, Cell Reports,
  % PMC4941620), but that requires GM-CSF *plus IL-4* to generate the IRF4+/Zbtb46+
  % cross-priming-competent subset -- and the GVAX depot supplies GM-CSF, not IL-4.
  % More decisively, GVAX/GM-CSF-vaccine CD8 priming is host-cDC1-dependent: it
  % collapses in Batf3-/- mice lacking cDC1 (Ferris 2022, PMID 35648641), which
  % means the direct (Batf3-independent) moDC route contributes little in vivo.
  % The old fraction (0.3) was an unanchored placeholder "chosen to enable a
  % non-zero contribution from the V_ID arm" -- so keeping it added a degenerate
  % parameter and double-routed cross-priming the literature attributes to handoff.
  % Re-add (with an IL-4 gate + submodel anchor) only if a motivating context arises.
  apcMigrationRxn = sbioselect(model,'Type','reaction','Name','mAPC_migration_ID_LN');
  if isempty(apcMigrationRxn)
      apcMigrationRxn = addreaction(model,'V_ID.mAPC -> V_LN.mcDC2');
      set(apcMigrationRxn,'Name','mAPC_migration_ID_LN');
  end
  set(apcMigrationRxn,'ReactionRate',...
      'k_APC_mig_ID_LN * V_ID.mAPC / (1 + V_LN.mcDC2 / K_APC_mig)');

  %% ---- Antigen transfer: migratory mcDC2 -> LN-resident cDC1 cross-licensing ----
  % GM-CSF-derived DC vaccines (GVAX) prime CD8 predominantly via antigen
  % handoff to LN-resident Batf3+ cDC1, not by direct cross-presentation
  % (Ferris 2022 PMID: 35648641). GMDC vaccines fail in Irf8+32-/- mice
  % lacking host cDC1. This reaction captures that route: antigen-loaded
  % mcDC2 arriving in LN drives resident cDC1 -> mcDC1 maturation.
  cDC1TransferRxn = sbioselect(model,'Type','reaction','Name','cDC1_Ag_transfer_LN');
  if isempty(cDC1TransferRxn)
      cDC1TransferRxn = addreaction(model,'V_LN.cDC1 -> V_LN.mcDC1');
      set(cDC1TransferRxn,'Name','cDC1_Ag_transfer_LN');
      set(cDC1TransferRxn,'Notes','Antigen transfer from GVAX-derived mcDC2 to LN-resident cDC1 (Ferris 2022). Catalytic by design: mcDC2 is the antigen donor and is not consumed (one migratory DC cross-dresses many resident cDC1). No antigen species moves in this reaction, so there is nothing to conserve here -- it is a count-gated cDC1->mcDC1 maturation; the antigen-LOAD dependence enters downstream at the H_P1-gated CD8 priming step (Teff_module). Bounded, not runaway: V_LN.cDC1 is the consumed substrate and is replenished only by its homeostatic source (APC_module), and the effective first-order rate k_cDC1_Ag_transfer*mcDC2 ~ 1e-3..1e-1/day stays well below cDC1 turnover. Audit m1-A11: no change.');
  end
  set(cDC1TransferRxn,'ReactionRate',...
      'k_cDC1_Ag_transfer * V_LN.cDC1 * V_LN.mcDC2');
end

%% ---- Optional: Cyclophosphamide for Treg depletion ----
if add_cy
    fprintf('   Adding cyclophosphamide for Treg depletion...\n');

    % Get compartments
    V_C = sbioselect(model, 'Name', 'V_C', 'Type', 'compartment');
    V_T = sbioselect(model, 'Name', 'V_T', 'Type', 'compartment');

    if ~isempty(V_C)
        % Add Cy species in central compartment (target for dosing)
        cy_species = sbioselect(model, 'Name', 'Cy', 'Type', 'species', 'Parent', V_C);
        if isempty(cy_species)
            cy_species = addspecies(V_C, 'Cy', 0, 'InitialAmountUnits', 'milligram');
            set(cy_species, 'Notes', 'Cyclophosphamide for Treg depletion (low-dose immunomodulatory)');
        end

        % Add durable PD effect-compartment species (Cy_eff): an indirect-
        % response state that builds from drug exposure and decays slowly, so
        % Treg suppression outlasts the parent-drug PK window to the d21 readout.
        cy_eff_species = sbioselect(model, 'Name', 'Cy_eff', 'Type', 'species', 'Parent', V_C);
        if isempty(cy_eff_species)
            cy_eff_species = addspecies(V_C, 'Cy_eff', 0, 'InitialAmountUnits', 'milligram');
            set(cy_eff_species, 'Notes', 'Cyclophosphamide durable PD effect compartment (mg-equiv signal; durable Treg-reconstitution block)');
        end

        % Add parameters
        k_Cy_clear = addparameter(model, 'k_Cy_clear', params.k_Cy_clear.Value, ...
            'ValueUnits', params.k_Cy_clear.Units, 'Notes', params.k_Cy_clear.Notes);
        k_Cy_Treg_deplete = addparameter(model, 'k_Cy_Treg_deplete', params.k_Cy_Treg_deplete.Value, ...
            'ValueUnits', params.k_Cy_Treg_deplete.Units, 'Notes', params.k_Cy_Treg_deplete.Notes);
        Cy_50_Treg = addparameter(model, 'Cy_50_Treg', params.Cy_50_Treg.Value, ...
            'ValueUnits', params.Cy_50_Treg.Units, 'Notes', params.Cy_50_Treg.Notes);
        k_Cy_eff_decay = addparameter(model, 'k_Cy_eff_decay', params.k_Cy_eff_decay.Value, ...
            'ValueUnits', params.k_Cy_eff_decay.Units, 'Notes', params.k_Cy_eff_decay.Notes);
        Emax_Cy_Treg = addparameter(model, 'Emax_Cy_Treg', params.Emax_Cy_Treg.Value, ...
            'ValueUnits', params.Emax_Cy_Treg.Units, 'Notes', params.Emax_Cy_Treg.Notes);
        k_Cy_on = addparameter(model, 'k_Cy_on', params.k_Cy_on.Value, ...
            'ValueUnits', params.k_Cy_on.Units, 'Notes', params.k_Cy_on.Notes);
        Cy_eff_50 = addparameter(model, 'Cy_eff_50', params.Cy_eff_50.Value, ...
            'ValueUnits', params.Cy_eff_50.Units, 'Notes', params.Cy_eff_50.Notes);

        % Cy clearance/degradation
        cyDegRxn = sbioselect(model, 'Type', 'reaction', 'Name', 'Cy_degradation');
        if isempty(cyDegRxn)
            cyDegRxn = addreaction(model, 'V_C.Cy -> null');
            set(cyDegRxn, 'Name', 'Cy_degradation');
            set(cyDegRxn, 'Notes', 'Cyclophosphamide clearance (t1/2 ~12-24 hours)');
        end
        set(cyDegRxn, 'ReactionRate', 'k_Cy_clear * V_C.Cy');

        % Cy_eff build-up: driven by drug exposure through the same Cy Hill, so
        % it accumulates while parent drug is present and persists afterward.
        cyEffBuildRxn = sbioselect(model, 'Type', 'reaction', 'Name', 'Cy_eff_buildup');
        if isempty(cyEffBuildRxn)
            cyEffBuildRxn = addreaction(model, 'null -> V_C.Cy_eff');
            set(cyEffBuildRxn, 'Name', 'Cy_eff_buildup');
            set(cyEffBuildRxn, 'Notes', 'Build-up of the durable Cy PD effect compartment from drug exposure');
        end
        set(cyEffBuildRxn, 'ReactionRate', 'k_Cy_on * (V_C.Cy / (Cy_50_Treg + V_C.Cy))');

        % Cy_eff decay: first-order; k_Cy_eff_decay sets the PD memory (duration).
        cyEffDecayRxn = sbioselect(model, 'Type', 'reaction', 'Name', 'Cy_eff_decay');
        if isempty(cyEffDecayRxn)
            cyEffDecayRxn = addreaction(model, 'V_C.Cy_eff -> null');
            set(cyEffDecayRxn, 'Name', 'Cy_eff_decay');
            set(cyEffDecayRxn, 'Notes', 'Decay of the durable Cy PD effect compartment (sets Treg-suppression duration)');
        end
        set(cyEffDecayRxn, 'ReactionRate', 'k_Cy_eff_decay * V_C.Cy_eff');

        % Cy-mediated Treg depletion in tumor (acute cytotoxic killing of cycling
        % Tregs while parent drug is present).
        if ~isempty(V_T)
            treg_species = sbioselect(model, 'Name', 'Treg', 'Type', 'species', 'Parent', V_T);
            if ~isempty(treg_species)
                cyTregDepRxn = sbioselect(model, 'Type', 'reaction', 'Name', 'Cy_Treg_depletion');
                if isempty(cyTregDepRxn)
                    cyTregDepRxn = addreaction(model, 'V_T.Treg -> null');
                    set(cyTregDepRxn, 'Name', 'Cy_Treg_depletion');
                    set(cyTregDepRxn, 'Notes', 'Selective Treg depletion by low-dose cyclophosphamide');
                end
                % Hill function: depletion increases with Cy concentration
                set(cyTregDepRxn, 'ReactionRate', ...
                    'k_Cy_Treg_deplete * V_T.Treg * (V_C.Cy / (Cy_50_Treg + V_C.Cy))');
            end

            % Durable reconstitution block: the Cy effect compartment throttles
            % tumor-local Treg proliferation (created in Treg_module) by up to
            % Emax_Cy_Treg. This is the mechanism that holds Tregs down to the
            % day-21 readout after parent Cy has cleared. Append the throttle to
            % the existing proliferation rate (house assembly pattern).
            for i = 1:length(model.reaction)
                if strcmp(model.reaction(i).reaction, 'null -> V_T.Treg') && ...
                        ~isempty(strfind(model.reaction(i).ReactionRate, 'k_Treg_pro_tumor')) %#ok<STREMP>
                    model.reaction(i).ReactionRate = [model.reaction(i).ReactionRate, ...
                        '*(1 - Emax_Cy_Treg*V_C.Cy_eff/(Cy_eff_50 + V_C.Cy_eff))'];
                end
            end
        end

        fprintf('   Cyclophosphamide module added successfully\n');
    else
        warning('Could not add cyclophosphamide: V_C compartment not found');
    end
end

end
