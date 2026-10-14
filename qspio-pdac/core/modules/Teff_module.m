% Teff Module
%
% Models Teff transport and activation by APCs [Use before antigen corresponding module]
%
% Inputs: model        -- SimBiology model object with four compartments
%         ID           -- T cell-antigen ID number [must be unique]
%         params       -- object containing model parameter Values, Units, and Notes:
%         cancer_types -- cell array of strings containing names of cancer types Teff kill
%
% Outputs: model -- SimBiology model object with new Tcell module


function model = Teff_module(model,params,cancer_types)

% Species Names
species_name = 'CD8';
antigen = 'P1';

% Add Species (final names — no rename pattern)
% Naive T cells
nCD8_C  = addspecies(model.Compartment(1),'nCD8',params.nCD8_C.Value/params.nCD8_div.Value,'InitialAmountUnits','cell');
    set(nCD8_C,'Notes',['Number of naive ' species_name ' cells in the central compartment']);
nCD8_P  = addspecies(model.Compartment(2),'nCD8',params.nCD8_P.Value/params.nCD8_div.Value,'InitialAmountUnits','cell');
    set(nCD8_P,'Notes',['Number of naive ' species_name ' cells in the peripheral compartment']);
nCD8_LN = addspecies(model.Compartment(4),'nCD8',params.nCD8_LN.Value/params.nCD8_div.Value,'InitialAmountUnits','cell', 'Notes', ['Number of naive ' species_name ' cells in the lymph node']);
% Activated T cells
aCD8 = addspecies(model.Compartment(4),'aCD8',0,'InitialAmountUnits','cell', 'Notes', ['Number of activated ' species_name ' cells in the lymph node']);
% Mature T cells
CD8_C = addspecies(model.Compartment(1),'CD8',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the central compartment']);
CD8_P = addspecies(model.Compartment(2),'CD8',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the peripheral compartment']);
CD8_T = addspecies(model.Compartment(3),'CD8',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the tumor compartment']);
CD8_LN = addspecies(model.Compartment(4),'CD8',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the lymph node compartment']);

% Add Parameters (final names — no rename pattern)
div_CD8 = addparameter(model,'div_CD8',params.nCD8_div.Value,'ValueUnits',params.nCD8_div.Units, 'Notes', 'T Cell Diversity ');
div_CD8.UserData.Source = params.nCD8_div.Notes;
    div_CD8.UserData.derivedFrom = params.nCD8_div.derivedFrom;
n_CD8_clones = addparameter(model,'n_CD8_clones',params.n_clones_tum.Value,'ValueUnits',params.n_clones_tum.Units, 'Notes', 'Number of tumor-specific T cell clones ');
n_CD8_clones.UserData.Source = params.n_clones_tum.Notes;
    n_CD8_clones.UserData.derivedFrom = params.n_clones_tum.derivedFrom;
p = addparameter(model,'q_nCD8_LN_in',params.q_nCD8_LN_in.Value,'ValueUnits',params.q_nCD8_LN_in.Units, 'Notes', 'Rate of naive T cell transport into the LN ');
p.UserData.Source = params.q_nCD8_LN_in.Notes;
    p.UserData.derivedFrom = params.q_nCD8_LN_in.derivedFrom;
p = addparameter(model,'q_CD8_LN_out',params.q_CD8_LN_out.Value,'ValueUnits',params.q_CD8_LN_out.Units, 'Notes', 'Rate of activated T cell transport out of the LN ');
p.UserData.Source = params.q_CD8_LN_out.Notes;
    p.UserData.derivedFrom = params.q_CD8_LN_out.derivedFrom;
p = addparameter(model,'q_nCD8_LN_out',params.q_nCD8_LN_out.Value,'ValueUnits',params.q_nCD8_LN_out.Units, 'Notes', 'Rate of naive T cell transport out of the LN ');
p.UserData.Source = params.q_nCD8_LN_out.Notes;
    p.UserData.derivedFrom = params.q_nCD8_LN_out.derivedFrom;
p = addparameter(model,'k_CD8_pro',params.k_CD8_pro.Value,'ValueUnits',params.k_CD8_pro.Units, 'Notes', [species_name ' proliferation rate']);
p.UserData.Source = params.k_CD8_pro.Notes;
    p.UserData.derivedFrom = params.k_CD8_pro.derivedFrom;
% Tumor-local CD8 proliferation params (antigen-driven, PD-1-suppressed)
p = addparameter(model,'k_CD8_T_pro',params.k_CD8_T_pro.Value,'ValueUnits',params.k_CD8_T_pro.Units, 'Notes', 'Tumor-local CD8 proliferation rate (antigen-gated, PD-1-suppressed)');
p.UserData.Source = params.k_CD8_T_pro.Notes;
    p.UserData.derivedFrom = params.k_CD8_T_pro.derivedFrom;
p = addparameter(model,'rho_CD8_max',params.rho_CD8_max.Value,'ValueUnits',params.rho_CD8_max.Units, 'Notes', 'Max CD8 density in tumor (carrying capacity for local proliferation)');
p.UserData.Source = params.rho_CD8_max.Notes;
    p.UserData.derivedFrom = params.rho_CD8_max.derivedFrom;
% Density set-point for the quadratic CD8 crowding-death term (V3 stabilizer, 2026-07-17).
p = addparameter(model,'rho_CD8_crowd',params.rho_CD8_crowd.Value,'ValueUnits',params.rho_CD8_crowd.Units, 'Notes', 'Density set-point for CD8 crowding death (crowding = linear death at this density)');
p.UserData.Source = params.rho_CD8_crowd.Notes;
    p.UserData.derivedFrom = params.rho_CD8_crowd.derivedFrom;
% IL-2-niche cap boost RETIRED 2026-06-22 (in-tumor restructure). The cd8_fc fix
% (alpha_IL2_cap*H_IL2_cap scaling of rho_CD8_max) was a workaround for dead division
% destiny: when N_aT was stuck ~4.7 the treated CD8 fold-change had no source, so the
% CAP was made treatment-responsive to manufacture it. Now the IL2_niche QSS fires
% division destiny (N_aT ~12) and fills the reservoir, so the treated fold comes from
% the real mechanism (more help -> more divisions -> more reservoir -> more influx).
% A trace confirmed CD8 sits at <9% of even the physical cap in both arms and the 3.8x
% treated fold occurred with H_IL2_cap saturated flat -- the cap contributes nothing.
% Cap reverts to the physical rho_CD8_max*V_T (a far safety ceiling for the
% super-critical k_CD8_T_pro tail). H_IL2_cap, alpha_IL2_cap, IL2_cap_50 removed.
% Mechanistic T cell activation parameters
p = addparameter(model,'k_T_activation_per_APC',params.k_T_activation_per_APC.Value,'ValueUnits',params.k_T_activation_per_APC.Units, 'Notes', 'Rate of naive T cell activation per mature APC');
p.UserData.Source = params.k_T_activation_per_APC.Notes;
    p.UserData.derivedFrom = params.k_T_activation_per_APC.derivedFrom;
% CD4-help / CD40L licensing of cDC1 (three-cell model: Bennett/Schoenberger/Ridge
% 1998; cDC1-specific Ferris 2020 PMID 32433613). A cognate CD4 helper licenses the
% cross-presenting cDC1 (CD40-CD40L -> CD70/IL-12 burst), raising its ability to
% PRIME CD8. Declared here as a dimensionless multiplier (default 1 = no-op so the
% module is self-consistent if Th_module is absent); the governing rule
%   H_CD4_help = f_help_indep + (1-f_help_indep)*V_LN.Th/(V_LN.Th+Th_50_help)
% is added in Th_module.m, which owns the V_LN.Th helper pool (mirrors the
% H_IL6_M2 declare-here / rule-in-fibroblast pattern; Teff runs before Th in
% immune_oncology_model_PDAC.m). f_help_indep is a small help-INDEPENDENT floor
% (innate DAMP/IL-12 cDC1 maturation already cross-primes some CD8 -> additive
% amplifier, NOT a hard required gate; same floor logic as the ArgI #4 fix).
% Applied ONLY to the CD8/mcDC1 priming branch below, never the Th/mcDC2 branch
% (helpers are the help). H_P1 already gates antigen; this adds the orthogonal
% help axis, so cold low-antigen tumors prime CD8 poorly via BOTH (bounded by the
% floor + Hill saturation -> no runaway).
H_CD4_help = addparameter(model,'H_CD4_help',1,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'CD4-help/CD40L licensing multiplier on cDC1->CD8 priming (rule supplied by Th_module)');
p = addparameter(model,'k_CD8_death',params.k_CD8_death.Value,'ValueUnits',params.k_CD8_death.Units, 'Notes', [species_name ' death rate']);
p.UserData.Source = params.k_CD8_death.Notes;
    p.UserData.derivedFrom = params.k_CD8_death.derivedFrom;
p = addparameter(model,'q_CD8_P_in',params.q_CD8_P_in.Value,'ValueUnits',params.q_CD8_P_in.Units, 'Notes', 'rate of CD8+ T cell transport into the peripheral compartment ');
p.UserData.Source = params.q_CD8_P_in.Notes;
    p.UserData.derivedFrom = params.q_CD8_P_in.derivedFrom;
p = addparameter(model,'q_CD8_P_out',params.q_CD8_P_out.Value,'ValueUnits',params.q_CD8_P_out.Units, 'Notes', 'rate of CD8+ T cell transport out of the peripheral compartment ');
p.UserData.Source = params.q_CD8_P_out.Notes;
    p.UserData.derivedFrom = params.q_CD8_P_out.derivedFrom;
p = addparameter(model,'q_CD8_T_in',params.q_CD8_T_in.Value,'ValueUnits',params.q_CD8_T_in.Units, 'Notes', 'rate of CD8+ T cell transport into the tumor compartment ');
p.UserData.Source = params.q_CD8_T_in.Notes;
    p.UserData.derivedFrom = params.q_CD8_T_in.derivedFrom;
p = addparameter(model,'q_nCD8_P_in',params.q_nCD8_P_in.Value,'ValueUnits',params.q_nCD8_P_in.Units, 'Notes', 'rate of naive CD8+ T cell transport into the peripheral compartment ');
p.UserData.Source = params.q_nCD8_P_in.Notes;
    p.UserData.derivedFrom = params.q_nCD8_P_in.derivedFrom;
p = addparameter(model,'q_nCD8_P_out',params.q_nCD8_P_out.Value,'ValueUnits',params.q_nCD8_P_out.Units, 'Notes', 'rate of naive CD8+ T cell transport out of the peripheral compartment ');
p.UserData.Source = params.q_nCD8_P_out.Notes;
    p.UserData.derivedFrom = params.q_nCD8_P_out.derivedFrom;
p = addparameter(model,'Q_nCD8_thym',params.Q_nCD8_thym.Value,'ValueUnits',params.Q_nCD8_thym.Units, 'Notes', 'Thymic output of naive CD8+ T cells');
p.UserData.Source = params.Q_nCD8_thym.Notes;
    p.UserData.derivedFrom = params.Q_nCD8_thym.derivedFrom;
p = addparameter(model,'k_nCD8_pro',params.k_nCD8_pro.Value,'ValueUnits',params.k_nCD8_pro.Units, 'Notes', 'rate of naive CD8+ T cell proliferation ');
p.UserData.Source = params.k_nCD8_pro.Notes;
    p.UserData.derivedFrom = params.k_nCD8_pro.derivedFrom;
K_nCD8_pro = addparameter(model,'K_nCD8_pro',params.K_nT_pro.Value,'ValueUnits',params.K_nT_pro.Units, 'Notes', 'half-maximal peripheral proliferation of naive CD8+ T cells');
K_nCD8_pro.UserData.Source = params.K_nT_pro.Notes;
    K_nCD8_pro.UserData.derivedFrom = params.K_nT_pro.derivedFrom;
p = addparameter(model,'k_nCD8_death',params.k_nT_death.Value,'ValueUnits',params.k_nT_death.Units, 'Notes', 'rate of naive CD8+ T cell death ');
p.UserData.Source = params.k_nT_death.Notes;
    p.UserData.derivedFrom = params.k_nT_death.derivedFrom;

k_C_CD8_exh = addparameter(model,'k_C_CD8_exh',params.k_Tcell.Value,'ValueUnits',params.k_Tcell.Units, 'Notes', 'Rate of T cell exhaustion by cancer cells ');
k_C_CD8_exh.UserData.Source = params.k_Tcell.Notes;
    k_C_CD8_exh.UserData.derivedFrom = params.k_Tcell.derivedFrom;

p = addparameter(model,'k_stiff_CD8_exh',params.k_stiff_CD8_exh.Value,'ValueUnits',params.k_stiff_CD8_exh.Units, 'Notes', 'Rate of stiffness-mediated T cell exhaustion');
p.UserData.Source = params.k_stiff_CD8_exh.Notes;
    p.UserData.derivedFrom = params.k_stiff_CD8_exh.derivedFrom;
% Baseline exhaustion parameters (chronic antigen exposure)
p = addparameter(model,'k_CD8_basal_exh',params.k_CD8_basal_exh.Value,'ValueUnits',params.k_CD8_basal_exh.Units, 'Notes', 'Baseline exhaustion rate from chronic TCR stimulation');
p.UserData.Source = params.k_CD8_basal_exh.Notes;
    p.UserData.derivedFrom = params.k_CD8_basal_exh.derivedFrom;
p = addparameter(model,'K_C_exh',params.K_C_exh.Value,'ValueUnits',params.K_C_exh.Units, 'Notes', 'Cancer cell number for half-maximal exhaustion induction');
p.UserData.Source = params.K_C_exh.Notes;
    p.UserData.derivedFrom = params.K_C_exh.derivedFrom;
% Exhausted T cell death rate (prevents indefinite accumulation)
p = addparameter(model,'k_CD8_exh_death',params.k_CD8_exh_death.Value,'ValueUnits',params.k_CD8_exh_death.Units, 'Notes', 'Death rate of exhausted CD8 T cells');
p.UserData.Source = params.k_CD8_exh_death.Notes;
    p.UserData.derivedFrom = params.k_CD8_exh_death.derivedFrom;
% Anti-PD-1 reinvigoration rate of progenitor-exhausted CD8 (mid-tier cA-C4)
p = addparameter(model,'k_reinvig',params.k_reinvig.Value,'ValueUnits',params.k_reinvig.Units, 'Notes', 'Max anti-PD-1 reinvigoration rate of progenitor-exhausted CD8 (at full PD-1 occupancy)');
p.UserData.Source = params.k_reinvig.Notes;
    p.UserData.derivedFrom = params.k_reinvig.derivedFrom;
% Progenitor-exhausted (Tpex) reservoir parameters (gateway exhaustion topology)
p = addparameter(model,'rho_pex_max',params.rho_pex_max.Value,'ValueUnits',params.rho_pex_max.Units, 'Notes', 'Niche carrying-capacity density for the progenitor-exhausted (Tpex) reservoir');
p.UserData.Source = params.rho_pex_max.Notes;
    p.UserData.derivedFrom = params.rho_pex_max.derivedFrom;
% p_pex_renew RETIRED (2026-06-17): the generative self-renewal term it scaled
% self-ignited the reservoir to the niche cap regardless of inflow, decoupling
% terminal exhaustion from its biological drivers. Reservoir is now inflow-sourced
% (the four CD8->CD8_pex exhaustion routes, niche-capped) with self-renewal folded
% into slow turnover (k_pex_death). See toy: /tmp/pex_loop_toy.py, pex_reservoir_toy.py.
p = addparameter(model,'k_pex_diff',params.k_pex_diff.Value,'ValueUnits',params.k_pex_diff.Units, 'Notes', 'Differentiation rate of Tpex -> terminal exhausted CD8');
p.UserData.Source = params.k_pex_diff.Notes;
    p.UserData.derivedFrom = params.k_pex_diff.derivedFrom;
p = addparameter(model,'k_pex_death',params.k_pex_death.Value,'ValueUnits',params.k_pex_death.Units, 'Notes', 'Death rate of progenitor-exhausted (Tpex) CD8 (~0.4x k_CD8_exh_death, Miller 2019)');
p.UserData.Source = params.k_pex_death.Notes;
    p.UserData.derivedFrom = params.k_pex_death.derivedFrom;
p = addparameter(model,'k_pex_reflux',params.k_pex_reflux.Value,'ValueUnits',params.k_pex_reflux.Units, 'Notes', 'Constitutive Tpex -> effector reflux rate (homeostatic stem-like replenishment)');
p.UserData.Source = params.k_pex_reflux.Notes;
    p.UserData.derivedFrom = params.k_pex_reflux.derivedFrom;
% H_pore_T and H_stiff_Texh are defined in fibroblast_module_enhanced.m
% (forward references resolved by SimBiology at simulation time)

% CXCL12-mediated T cell exclusion (PDAC-specific)
H_CXCL12_Texcl = addparameter(model,'H_CXCL12_Texcl',0,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function for CXCL12-mediated T cell exclusion');

% Vascular-access gating of blood->tumor extravasation (PDAC hypovascularity).
% V_T.K_vasc (vasculature-dependent carrying capacity, from cancer_module) grows
% slower than V_T as the tumor progresses, so vascular density V_T.K_vasc/V_T falls.
% H_vasc multiplies the per-volume extravasation rate of every blood->tumor
% recruitment reaction (CD8 baseline here; Th baseline in Th_module).
% vasc_ratio is a dimensionless intermediate so the ^n_vasc
% term doesn't trip SimBiology dimensional analysis (same pattern as pO2_ratio).
p = addparameter(model,'K50_vasc',params.K50_vasc.Value,'ValueUnits',params.K50_vasc.Units, 'Notes', 'EC50 for vascular-access gating of extravasation (V_T.K_vasc/V_T)');
p.UserData.Source = params.K50_vasc.Notes;
    p.UserData.derivedFrom = params.K50_vasc.derivedFrom;
p = addparameter(model,'n_vasc',params.n_vasc.Value,'ValueUnits',params.n_vasc.Units, 'Notes', 'Hill coefficient for vascular-access gating of extravasation');
p.UserData.Source = params.n_vasc.Notes;
    p.UserData.derivedFrom = params.n_vasc.derivedFrom;
addparameter(model,'vasc_ratio',0,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Vascular density (V_T.K_vasc/V_T) normalized by K50_vasc (dimensionless intermediate)');
addrule(model,'vasc_ratio = (V_T.K_vasc/V_T)/K50_vasc','repeatedAssignment');
H_vasc = addparameter(model,'H_vasc',0,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Vascular-access Hill: per-volume extravasation efficiency, falls as the tumor outgrows its vasculature');
addrule(model,'H_vasc = vasc_ratio^n_vasc/(1 + vasc_ratio^n_vasc)','repeatedAssignment');

% k_Treg_CD8_exh is already in the model (created by Treg_module)

% Antigen Default Hill Function
p = addparameter(model,['H_' antigen],1,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', ['Hill function of tumor antigen ' antigen]);

% Add Reactions
% Thymic output of naive T cell
reaction = addreaction(model,'null -> V_C.nCD8');
    set(reaction,'ReactionRate','Q_nCD8_thym/div_CD8');
    set(reaction,'Notes','Thymic output of naive T cell to blood');
% Naive T cell proliferation in the peripheral compartment and the LN compartment
reaction = addreaction(model,'null -> V_P.nCD8');
    set(reaction,'ReactionRate','k_nCD8_pro/div_CD8*V_P.nCD8/(K_nCD8_pro/div_CD8+V_P.nCD8)');
    set(reaction,'Notes','Naive T cell proliferation in the peripheral compartment');
reaction = addreaction(model,'null -> V_LN.nCD8');
    set(reaction,'ReactionRate','k_nCD8_pro/div_CD8*V_LN.nCD8/(K_nCD8_pro/div_CD8+V_LN.nCD8)');
    set(reaction,'Notes','Naive T cell proliferation in the TDLN compartment');
% Naive T cell death
reaction = addreaction(model,'V_P.nCD8 -> null');
    set(reaction,'ReactionRate','k_nCD8_death*V_P.nCD8');
    set(reaction,'Notes','Naive T cell death in the peripheral compartment');
reaction = addreaction(model,'V_C.nCD8 -> null');
    set(reaction,'ReactionRate','k_nCD8_death*V_C.nCD8');
    set(reaction,'Notes','Naive T cell death in the central compartment');
reaction = addreaction(model,'V_LN.nCD8 -> null');
    set(reaction,'ReactionRate','k_nCD8_death*V_LN.nCD8');
    set(reaction,'Notes','Naive T cell death in the TDLN compartment');
% Naive T cell transport into and out of the peripheral compartment
reaction = addreaction(model,'V_C.nCD8 -> V_P.nCD8');
    set(reaction,'ReactionRate','q_nCD8_P_in*V_C.nCD8');
    set(reaction,'Notes','Naive T cell entry into the peripheral compartment');
reaction = addreaction(model,'V_P.nCD8 -> V_C.nCD8');
    set(reaction,'ReactionRate','q_nCD8_P_out*V_P.nCD8');
    set(reaction,'Notes','Naive T cell exit from the peripheral compartment');
% Naive T cell transport into and out of the lymph node
reaction = addreaction(model,'V_C.nCD8 -> V_LN.nCD8');
    set(reaction,'ReactionRate','q_nCD8_LN_in*V_C.nCD8');
    set(reaction,'Notes','Naive T cell entry into the lymph node');
reaction = addreaction(model,'V_LN.nCD8 -> V_C.nCD8');
    set(reaction,'ReactionRate','q_nCD8_LN_out*V_LN.nCD8');
    set(reaction,'Notes','Naive T cell exit from the lymph node');

% Naive T cell activation - mechanistic clonal expansion model
reaction = addreaction(model,'V_LN.nCD8 -> null');
    set(reaction,'ReactionRate',['k_T_activation_per_APC*V_LN.mcDC1*H_' antigen '*V_LN.nCD8*H_CD4_help']);
    set(reaction,'Notes','Naive CD8 T cell activation by mature cDC1 (cross-presentation), licensed by CD4 help (H_CD4_help)');
reaction = addreaction(model,'null -> V_LN.aCD8');
    set(reaction,'ReactionRate',['k_T_activation_per_APC*V_LN.mcDC1*H_' antigen '*V_LN.nCD8*H_CD4_help*n_CD8_clones']);
    set(reaction,'Notes','Naive CD8 -> activated CD8 commitment (x clone diversity). Clonal EXPANSION is carried solely by the 2^N_aT proliferation cascade below (N_aT = N_div_base + N_div_costim*H_CD28 + N_IL2_CD8*IL2hill, Marchingo 2014). The former *n_T_expansion_burst fold-factor here double-counted that expansion (audit V12; the Butz1998 submodel forward model itself sums log2(burst)+N_aT) and was removed.');
% Activated T cell proliferation
reaction = addreaction(model,'V_LN.aCD8 -> null');
    set(reaction,'ReactionRate','k_CD8_pro/N_aT*V_LN.aCD8');
    set(reaction,'Notes',['a' species_name ' cell proliferation']);
reaction = addreaction(model,'null -> V_LN.CD8');
    set(reaction,'ReactionRate','k_CD8_pro/N_aT*2^(N_aT)*V_LN.aCD8');
    set(reaction,'Notes',['a' species_name ' cell proliferation']);

% T cell Death
reaction = addreaction(model,'V_C.CD8 -> null');
    set(reaction,'ReactionRate','k_CD8_death*V_C.CD8');
    set(reaction,'Notes','T cell death in the central compartment');
reaction = addreaction(model,'V_P.CD8 -> null');
    set(reaction,'ReactionRate','k_CD8_death*V_P.CD8');
    set(reaction,'Notes','T cell death in the peripheral compartment');
reaction = addreaction(model,'V_LN.CD8 -> null');
    set(reaction,'ReactionRate','k_CD8_death*V_LN.CD8');
    set(reaction,'Notes','T cell death in the lymph node compartment');
% NOTE (2026-06-22, in-tumor restructure): tumoral CD8 linear death is RESTORED
% (it was removed 2026-06-18 to make the cap bind). The June removal was a numerical
% convenience, not biology: death and exhaustion are distinct fates (a live effector
% can die OR go dysfunctional), so both belong. Restoring death converts the level
% from cap-set (rho_CD8_max-pinned, over-dispersed because the sigma=2 k_CD8_T_pro
% prior straddles the prolif=exhaustion bifurcation and X* diverges as prolif->exh)
% to an INFLUX/TURNOVER balance X* = influx/(k_CD8_death + exhaustion - prolif).
% Restoring death raises the bifurcation threshold from exhaustion (~0.13/day,
% backed out of the baseline trace) to k_CD8_death+exhaustion (~0.23/day), cutting
% the cap-slam tail ~32%->22% of draws, AND keeps the balance denominator safely
% off zero (~0.18 vs ~0.08) so X* no longer diverges near the bifurcation -- that is
% the cd8_pct/cd4_pct over-dispersion fix. The physical cap rho_CD8_max*V_T is kept
% as a far safety ceiling for the residual super-critical tail (CD8 sits <9% of it).
% COUPLING: death ~2.3x lowers X*, so the influx side (door via V_C.CD8 / q_CD8_T_in,
% seed) must rise in step to hold cd8_density -- tracked in the restructure plan.
% k_CD8_death keeps its De Boer & Perelson 1995 anchor; k_CD8_T_pro keeps Blessin.
% Tumor-local CD8 proliferation (PDAC TME-driven, capped by rho_CD8_max).
% Captures in-tumor TIL clonal expansion. Rate k_CD8_T_pro is anchored by the
% Blessin 2021 PDAC Ki67=12% submodel target — a *realized* in-vivo proliferation
% rate already gated by antigen availability, cytokine niche, and exhaustion in
% the actual PDAC TME. This rate is deliberately NOT gated again by H_P1 (the
% aggregate pMHC-TCR Hill, also used for LN naive priming): gating it would
% double-count antigen availability against the empirical Ki67 anchor, which
% already reflects in-vivo antigen exposure. (Historical note: an earlier
% revision justified this de-gating by claiming H_P1 ~ 0.07 throughout under a
% submodel-anchored p1_50; that is stale — p1_50 is no longer submodel-anchored
% and H_P1 now sits ~0.58 at the posterior. The de-gating still holds on the
% Ki67-double-counting argument, independent of the H_P1 magnitude.)
% V_T.CD8 here is the live (non-exhausted) pool — exhausted cells partition into
% V_T.CD8_exh via existing routes, so (1-H_PD1_C1) gating would also double-count.
% PD-1+ TIL proliferate readily (Ki67+); PD-1 suppresses killing, not division.
% Post-antigen-clearance, the k_cell_clear × K_C2_recruit/(C^2+K_C2_recruit)
% route drains V_T.CD8 — that, not H_P1, is the correct healthy-state attractor.
reaction = addreaction(model,'null -> V_T.CD8');
    set(reaction,'ReactionRate','k_CD8_T_pro*V_T.CD8/(1 + V_T.CD8/(rho_CD8_max*V_T + cell))');
    set(reaction,'Notes','Tumor-local CD8 proliferation, SATURATING in density (Blessin-anchored k_pro; rho_CD8_max*V_T is the half-saturation density, NOT a hard ceiling). Reformulated 2026-07-13 from the logistic k_pro*CD8*(1 - CD8/cap): see notes/model/tcell_carrying_capacity_inversion.md. The logistic has no max(0,.) clamp, so once GVAX drove CD8 past the cap the "proliferation" reaction ran as a QUADRATIC SINK whose strength scaled with k_pro -- raising CD8 proliferation destroyed CD8 (4x k_CD8_T_pro -> 0.775x CD8), and influx sensitivity was crushed to a square root. The pools sit AT the cap boundary (d21 CD8/cap ~1.1 at the submodel-anchored theta), so the sign flipped exactly where the model lives. The saturating form is monotone in k_pro and in influx at every density, so behaviour no longer depends on which side of the cap a draw lands. The pool is bounded by k_CD8_death (restored 2026-06-22), not by the cap. Influx/turnover sets the level -- as this comment always claimed, but which the logistic did not deliver.');
% Tumor-local CD8 homeostatic death (RESTORED 2026-06-22 in-tumor restructure).
% Ordinary effector apoptosis/AICD, distinct from exhaustion (which is a separate
% fate routing to CD8_pex/CD8_exh). Mirrors the circulating-compartment CD8 death
% (V_C/V_P/V_LN) and shares the k_CD8_death De Boer & Perelson 1995 anchor. This is
% the +k_CD8_death term in the influx/turnover balance that sets the in-tumor level.
reaction = addreaction(model,'V_T.CD8 -> null');
    set(reaction,'ReactionRate','k_CD8_death*V_T.CD8*(1 + V_T.CD8/(rho_CD8_crowd*V_T + cell))');
    set(reaction,'Notes','Tumor-local homeostatic CD8 death with a density-dependent CROWDING term (V3 stabilizer, 2026-07-17). Linear apoptosis/AICD k_CD8_death*CD8 plus a quadratic crowding death k_CD8_death*CD8^2/(rho_CD8_crowd*V_T) representing cytokine/nutrient competition (per-cell survival drops as local CD8 density rises). The crowding term is NOT meant to lower the target CD8 level -- it removes the near-critical runaway. With linear birth vs linear death the in-tumor CD8 balance sits at a bifurcation (below it CD8 collapses, above it the coupled tumour/immune loop diverges to NaN; ~30% of prior draws blow up and the ~62 cells/mm^2 target lives on that edge). The C^2 death bounds the high-density branch, converting the explode side into a finite steady state so the high-density regime becomes reachable. Crowding equals the linear death at density = rho_CD8_crowd (2e7 cell/mL ~ 220 cells/mm^2), ~28% extra death at the target. Distinct from exhaustion routing.');
% k_CD8_seed REMOVED 2026-07-13. A cancer-gated constitutive CD8 homing flux was added here in
% v24 (732b9f6) to stop ~1/3 of virtual patients collapsing to an unphysical ~0 CD8 -- V_T.CD8=0
% is a fixed point of the proportional effector+Tpex cascade, and draws with k_CD8_T_pro below
% the ~0.15/day exhaustion-drain bifurcation free-fell into it. That was a band-aid on a SUPPLY
% bug: 7c0366d fixed the supply chain properly (k_T_activation_per_APC bound to Bousso, residence
% re-anchor) and took cd8_density from 0.005 to 21 /mm^2, so the
% collapse branch no longer exists and the seed was commented out. The parameter was nonetheless
% left registered at Value=0 and never entered pdac_priors.csv, so it rode into the SBML/C++ as a
% dead constant with no consumers. Parameter and reaction are now both gone. The CD4 analog
% k_Th_seed (Th_module.m) is deliberately RETAINED and live -- the cd4_pct bistability it guards
% against is still unfixed. See notes/calibration/d21_composition_tension_diagnosis.md.
% Baseline T cell exhaustion from chronic antigen exposure (PDL1-independent route).
% Gated by (1-H_PD1_C1) to avoid additive double-counting with the PDL1-mediated
% chronic-antigen exhaustion route below (k_C_CD8_exh × H_PD1_C1). Both routes
% are driven by the same chronic-antigen pool; the PDL1-dependent mechanism
% should *replace*, not add to, the PDL1-independent baseline as PD-1:PDL1
% engagement increases. (Inverse of the proliferation-side reasoning at lines
% 204-207 above, where gating by (1-H_PD1_C1) would *cause* double-counting.)
reaction = addreaction(model,'V_T.CD8 -> V_T.CD8_pex');
    set(reaction,'ReactionRate','k_CD8_basal_exh*V_T.CD8*C_total/(C_total+K_C_exh+cell)*(1-H_PD1_C1)*(1 - V_T.CD8_pex/(rho_pex_max*V_T + cell))');
    set(reaction,'Notes','Baseline T cell exhaustion from chronic antigen exposure (PDL1-indep; gated by 1-H_PD1_C1 to avoid double-counting with route 4 below). Gateway: enters the progenitor-exhausted (Tpex) reservoir, not terminal.');

% T cell clearance upon Ag clearance
reaction = addreaction(model,'V_T.CD8 -> V_T.CD8_exh');
    set(reaction,'ReactionRate','k_cell_clear*V_T.CD8*(K_C2_recruit/(C_total^2 + K_C2_recruit))');
    set(reaction,'Notes','T cell clearance upon antigen clearance');

% T cell exhaustion from Treg (Treg-mediated suppression -> exhausted phenotype)
reaction = addreaction(model,'V_T.CD8 -> V_T.CD8_pex');
    set(reaction,'ReactionRate','k_Treg_CD8_exh*V_T.CD8*Tregs_/(V_T.CD8+Tregs_+cell)*(1 - V_T.CD8_pex/(rho_pex_max*V_T + cell))');
    set(reaction,'Notes','T cell exhaustion from Tregs (gateway: enters Tpex reservoir)');
% T cell exhaustion from cancer (PD-1/PD-L1 mediated)
reaction = addreaction(model,'V_T.CD8 -> V_T.CD8_pex');
    set(reaction,'ReactionRate','k_C_CD8_exh*V_T.CD8*C_total/(C_total+V_T.CD8+cell)*H_PD1_C1*(1 - V_T.CD8_pex/(rho_pex_max*V_T + cell))');
    set(reaction,'Notes','T cell exhaustion from cancer (PD-1 mediated; gateway: enters Tpex reservoir)');
% T cell exhaustion from ECM (desmoplasia-induced)
reaction = addreaction(model,'V_T.CD8 -> V_T.CD8_pex');
    set(reaction,'ReactionRate','k_stiff_CD8_exh*V_T.CD8*H_stiff_Texh*(1 - V_T.CD8_pex/(rho_pex_max*V_T + cell))');
    set(reaction,'Notes','T cell exhaustion from ECM (gateway: enters Tpex reservoir)');

% Terminal exhausted T cell death (prevents indefinite accumulation)
reaction = addreaction(model,'V_T.CD8_exh -> null');
    set(reaction,'ReactionRate','k_CD8_exh_death*V_T.CD8_exh');
    set(reaction,'Notes','Death of terminally exhausted CD8 T cells');

% --- Progenitor-exhausted (Tpex) reservoir dynamics (INFLOW-SOURCED topology) ---
% Effector CD8 exhaust into V_T.CD8_pex (the four exhaustion routes above, each now
% niche-capped by (1 - CD8_pex/(rho_pex_max*V_T))), which differentiates to terminal
% CD8_exh, turns over slowly (k_pex_death folds in self-renewal as persistence), and
% refluxes to the effector pool. REFORMULATED 2026-06-17: the previous generative
% self-renewal (null -> CD8_pex at p_pex_renew*P*(1-P/cap), net-positive vs outflow)
% self-ignited the reservoir to the niche cap regardless of effector inflow. That
% pinned terminal exhaustion to rho_pex_max*k_pex_diff and DECOUPLED it from the
% Treg/cancer/stiffness/chronic-Ag routes that should drive it (toy: 1000x inflow
% sweep moved terminal exhausted only ~5.7x; cap sweep moved it linearly). The niche
% now caps the INFLOW (Jansen 2019 niche capacity preserved as a ceiling, not a
% self-filling attractor); the reservoir is sustained by exhaustion inflow, so
% cd8exh re-couples to exhaustion biology. Effector setpoint stability is carried by
% the effector's own logistic cap (rho_CD8_max) + trafficking inflow, not by a
% reservoir reflux floor (verified in /tmp/pex_loop_toy.py: effector smooth vs k_pro,
% no knife-edge). See notes + feedback_misses_are_formulation_not_priors.
% Tpex -> terminal exhausted differentiation
reaction = addreaction(model,'V_T.CD8_pex -> V_T.CD8_exh');
    set(reaction,'ReactionRate','k_pex_diff*V_T.CD8_pex');
    set(reaction,'Notes','Differentiation of progenitor-exhausted (Tpex) to terminal exhausted CD8');
% Tpex death (slower than terminal; Miller 2019 Fig 3f progenitor ~0.4x terminal)
reaction = addreaction(model,'V_T.CD8_pex -> null');
    set(reaction,'ReactionRate','k_pex_death*V_T.CD8_pex');
    set(reaction,'Notes','Death of progenitor-exhausted (Tpex) CD8 (~0.4x terminal, Miller 2019)');
% Constitutive (therapy-independent) Tpex -> effector reflux: stem-like progenitors
% homeostatically replenish the functional effector pool even without checkpoint
% blockade (Siddiqui 2019). Bounded by the effector carrying cap (rho_CD8_max in the
% proliferation reaction), so it cannot run away. The anti-PD-1 k_reinvig arm above
% boosts this same progenitor->effector transition under therapy.
reaction = addreaction(model,'V_T.CD8_pex -> V_T.CD8');
    set(reaction,'ReactionRate','k_pex_reflux*V_T.CD8_pex');
    set(reaction,'Notes','Constitutive Tpex -> effector reflux (homeostatic stem-like replenishment of functional CD8)');
% Anti-PD-1 reinvigoration (reflux arm): ONLY progenitor-exhausted (Tpex) cells are
% rescued back to the effector pool; terminal CD8_exh cannot be reinvigorated
% (Miller/Siddiqui 2019). Gated by anti-PD-1 occupancy (phi_aPD1_C1, =0 without nivo),
% so this arm is INERT at baseline -- it does not affect the diagnosis-state CD8 count.
% NOTE: the k_reinvig submodel anchor (k_reinvig_PDAC_deriv001) was derived from a
% competition identity assuming the OLD lumped-pool topology (f = k_reinvig/(k_reinvig
% +k_CD8_exh_death)); with Tpex now an explicit pool that mapping must be revisited in
% the phase-2 checkpoint-coupling pass.
reaction = addreaction(model,'V_T.CD8_pex -> V_T.CD8');
    set(reaction,'ReactionRate','k_reinvig*V_T.CD8_pex*phi_aPD1_C1');
    set(reaction,'Notes','Anti-PD-1 reinvigoration: progenitor-exhausted (Tpex) CD8 reflux to effector pool (terminal cells not rescued; cA-C4)');

% T cell transport
% Central & Peripheral
reaction = addreaction(model,'V_C.CD8 -> V_P.CD8');
    set(reaction,'ReactionRate','q_CD8_P_in*V_C.CD8');
    set(reaction,'Notes','T cell transport into the peripheral compartment');
reaction = addreaction(model,'V_P.CD8 -> V_C.CD8');
    set(reaction,'ReactionRate','q_CD8_P_out*V_P.CD8');
    set(reaction,'Notes','T cell transport out of the peripheral compartment');
% Central & tumor (with CXCL12-mediated exclusion + vascular-access gating for PDAC)
reaction = addreaction(model,'V_C.CD8 -> V_T.CD8');
    set(reaction,'ReactionRate','q_CD8_T_in*V_T*V_C.CD8*(C_total^2/(C_total^2 + K_C2_recruit))*(1-H_CXCL12_Texcl)*H_vasc');
    set(reaction,'Notes','T cell transport into the tumor compartment (reduced by CXCL12-mediated exclusion and vascular-access gating H_vasc). NOTE: the desmoplastic pore barrier (1-H_pore_T) is deliberately NOT applied here — V_T is the whole tumor region including stroma, and V_T.CD8 (matching the observed cd8_density) counts stromal CD8. PDAC immune exclusion keeps CD8 in the stroma but away from cancer nests, which is represented on the KILL rate (1-H_pore_T in k_C_Tcell_eff), not on infiltration into the region.');
% Central & LN
reaction = addreaction(model,'V_LN.CD8 -> V_C.CD8');
    set(reaction,'ReactionRate','q_CD8_LN_out*V_LN.CD8');
    set(reaction,'Notes','T cell transport out of the lymph node compartment');

% ========================================================================
% TUMOR-LOCAL NAIVE CD8 PRIMING (matches ABM dendritic_cell.cuh + t_cell.cuh)
% ========================================================================
% Naive CD8 T cells recruited to tumor encounter mature cDC1s in situ and
% are primed to effectors. Same per-APC mass-action rate as LN priming
% (k_T_activation_per_APC). No H_P1 or H_CD4_help gating — tumor-local DCs
% already matured (requiring antigen), and CD4 licensing is an LN three-cell
% model (Bennett 1998); in-tumor priming is direct contact (ABM: t_cell.cuh
% lines 179-209). Antibody boost on DC maturation is added by Bcell_module
% assembly-time rewrite (H_HMGB1 -> boosted form in APC_module reactions).

% Naive CD8 in tumor
addspecies(model.Compartment(3),'nCD8',0,'InitialAmountUnits','cell', 'Notes', 'Naive CD8 T cells in tumor (recruited from blood, primed by local mature cDC1)');

% Trafficking: blood naive CD8 -> tumor (same extravasation as effector CD8)
reaction = addreaction(model,'V_C.nCD8 -> V_T.nCD8');
    set(reaction,'ReactionRate','q_CD8_T_in*V_T*V_C.nCD8*(C_total^2/(C_total^2 + K_C2_recruit))*(1-H_CXCL12_Texcl)*H_vasc');
    set(reaction,'Notes','Naive CD8 trafficking to tumor (same extravasation gating as effector CD8)');

% Tumor-local priming: V_T.nCD8 consumed by mature cDC1
reaction = addreaction(model,'V_T.nCD8 -> null');
    set(reaction,'ReactionRate','k_T_activation_per_APC*V_T.mcDC1*V_T.nCD8');
    set(reaction,'Notes','Naive CD8 consumed by tumor-local cDC1 priming');
% Production: clonal burst (n_CD8_clones effectors per naive consumed)
reaction = addreaction(model,'null -> V_T.CD8');
    set(reaction,'ReactionRate','k_T_activation_per_APC*V_T.mcDC1*V_T.nCD8*n_CD8_clones');
    set(reaction,'Notes','Tumor-local CD8 priming by mature cDC1 (clonal burst; ABM: PARAM_DC_PRIME_K_CD8)');

% Naive CD8 death in tumor (same rate as systemic naive death)
reaction = addreaction(model,'V_T.nCD8 -> null');
    set(reaction,'ReactionRate','k_nCD8_death*V_T.nCD8');
    set(reaction,'Notes','Naive CD8 death in tumor');
% Antigen-clearance funneling
reaction = addreaction(model,'V_T.nCD8 -> null');
    set(reaction,'ReactionRate','k_cell_clear*V_T.nCD8*(K_C2_recruit/(C_total^2 + K_C2_recruit))');
    set(reaction,'Notes','Naive CD8 clearance upon antigen clearance');

% Get Model Rules for Updating
model_rules = get(model,'Rules');

% Update Total T Cells in tumor (Rule 3) — includes V_T.nCD8
Tcell_rule = model_rules(3);
rule = get(Tcell_rule,'Rule');
set(Tcell_rule,'Rule',[rule '+V_T.' species_name '+V_T.nCD8']);

% Update Total T Cells in LN (Rule 4)
Tcell_rule = model_rules(4);
rule = get(Tcell_rule,'Rule');
set(Tcell_rule,'Rule',[rule '+V_LN.' species_name]);

% R50_Treg: Treg:Teff ratio for 50% inhibition of killing (PRIMARY calibration parameter)
R50_Treg = addparameter(model,'R50_Treg',params.R50_Treg.Value,'ValueUnits',params.R50_Treg.Units, 'Notes', 'Treg:Teff ratio for 50% inhibition of T cell killing');
R50_Treg.UserData.Source = params.R50_Treg.Notes;
    R50_Treg.UserData.derivedFrom = {params.R50_Treg.Notes};

% K_T_Treg: Derived from R50_Treg for use in equations (K_T_Treg = 1/R50_Treg)
K_T_Treg = addparameter(model,'K_T_Treg',params.K_T_Treg.Value,'ValueUnits',params.K_T_Treg.Units,'ConstantValue',false, 'Notes', 'Treg inhibition factor (derived: 1/R50_Treg)');
K_T_Treg.UserData.Source = params.K_T_Treg.Notes;
    K_T_Treg.UserData.derivedFrom = params.K_T_Treg.derivedFrom;
addrule(model, 'K_T_Treg = 1/R50_Treg', 'repeatedAssignment');
% Mechanistic tumor killing parameters
p = addparameter(model,'v_T_search_volume',params.v_T_search_volume.Value,'ValueUnits',params.v_T_search_volume.Units, 'Notes', 'Volume searched by one T cell per day');
p.UserData.Source = params.v_T_search_volume.Notes;
    p.UserData.derivedFrom = params.v_T_search_volume.derivedFrom;
p = addparameter(model,'p_T_kill_per_contact',params.p_T_kill_per_contact.Value,'ValueUnits',params.p_T_kill_per_contact.Units, 'Notes', 'Kill probability per T cell-tumor contact');
p.UserData.Source = params.p_T_kill_per_contact.Notes;
    p.UserData.derivedFrom = params.p_T_kill_per_contact.derivedFrom;
% Emax_TGFb_Teff: max fractional TGFb suppression of killing (partial antagonist).
% Floors the TGFb gate so saturating TGFb suppresses at most Emax_TGFb_Teff (~40%,
% Ahmadzadeh 2005 Fig 4C plateau) rather than driving killing to zero.
p = addparameter(model,'Emax_TGFb_Teff',params.Emax_TGFb_Teff.Value,'ValueUnits',params.Emax_TGFb_Teff.Units, 'Notes', 'Max fractional TGFb suppression of CD8 killing (Ahmadzadeh 2005 plateau)');
p.UserData.Source = params.Emax_TGFb_Teff.Notes;
    p.UserData.derivedFrom = params.Emax_TGFb_Teff.derivedFrom;

% Effective T cell killing rate (computed parameter)
k_C_Tcell_eff = addparameter(model,'k_C_Tcell_eff',0,'ValueUnits','1/day','ConstantValue',false, 'Notes', 'Effective T cell killing rate including all inhibition factors');

% Add repeated assignment rule for effective T cell killing rate
% TODO: k_C_Tcell_eff conflates spatial access (pore exclusion), functional state
% (Treg/TGFb suppression), and synapse-level inhibition (PD-1) into one rate.
% Future: decompose into infiltration flux × contact rate × per-contact kill probability.
tcellEffRuleTxt = 'k_C_Tcell_eff = p_T_kill_per_contact * v_T_search_volume * V_T.CD8 / V_T / cell * (1-H_pore_T) * V_T.CD8/(V_T.CD8+K_T_Treg*Tregs_+cell) * (1-H_PD1_C1) * (1-Emax_TGFb_Teff*H_TGFb_Teff)';

% Get existing rules to check for duplicates
R = sbioselect(model,'Type','rule');
didReplaceTcellEff = false;
for k = 1:numel(R)
    if contains(strrep(R(k).Rule,' ',''), 'k_C_Tcell_eff=')
        set(R(k),'Rule',tcellEffRuleTxt); didReplaceTcellEff = true;
    end
end
if ~didReplaceTcellEff
    addrule(model, tcellEffRuleTxt, 'repeatedAssignment');
end

% Update Cancer Killing by T Cells (Rule 5)
if (exist('cancer_types','var'))
    for i = 1:length(cancer_types)
        reaction = addreaction(model,['V_T.' cancer_types{i} ' -> V_T.C_x']);
            set(reaction,'ReactionRate',['k_C_Tcell_eff*V_T.' cancer_types{i}]);
            set(reaction,'Notes','Cancer cell killing by T cells');
        rule = get(model_rules(5),'Rule');
            set(model_rules(5),'Rule',[rule,'+(k_' cancer_types{i} '_death+k_' cancer_types{i} '_therapy)*V_T.' cancer_types{i} '+k_C_Tcell_eff*V_T.' cancer_types{i}]);
    end
end

warning('off','SimBiology:DimAnalysisNotDone_MatlabFcn_Dimensionless');
