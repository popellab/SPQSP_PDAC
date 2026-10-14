% Helper T Cell Module
%
% Models Th activation and transport
%
% Inputs: model        -- SimBiology model object with four compartments
%         params       -- object containing the default parameters
%
% Outputs: model -- SimBiology model object with new Th module


function model = Th_module(model,params)

% nCD4 is the shared naive CD4+ pool (consumed by both Th and Treg activation)
species_name = 'Th';

% Add Species (final names — no rename pattern)
% Activated T cells
aTh = addspecies(model.Compartment(4),'aTh',0,'InitialAmountUnits','cell', 'Notes', ['Number of activated ' species_name ' cells in the lymph node']);
% Mature T cells
Th_C = addspecies(model.Compartment(1),'Th',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the central compartment']);
Th_P = addspecies(model.Compartment(2),'Th',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the peripheral compartment']);
Th_T = addspecies(model.Compartment(3),'Th',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the tumor compartment']);
Th_LN = addspecies(model.Compartment(4),'Th',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the lymph node compartment']);

% Th priming uses the mechanistic mass-action formulation (parallel to CD8
% priming in Teff_module.m, refactored in commit 72d2cfd Nov 2025 from the
% old H_APC_Th saturating Hill). Naive CD4 activation rate is proportional
% to mature cDC2 abundance — no longer Hill-saturated. Anchored on Miller
% 2004 (T cell scanning rate in LN, PMID 15240696). Reuses k_T_activation_per_APC
% from CD8 priming since LN T-cell scanning kinetics are comparable for CD4 vs
% CD8 (within a 2-3× factor SBI can resolve). Clonal expansion is carried by the
% 2^N_aTh proliferation cascade (Marchingo 2014); the former n_T_expansion_burst
% fold-factor on the activation step double-counted it (audit V12) and was
% removed. Old k_Th_act and H_APC_Th retired from Th priming
% (Treg priming still uses k_Treg_act + H_APC_Treg by design — natural
% Treg generation is APC-saturated, not antigen-amplified).
% Audit m1-A20 (adjudicated 2026-06-05, ACCEPTED as-is): this mass-action effector
% form is unbounded in mcDC2 (a GVAX trace confirmed V_LN.nCD4 does NOT self-limit),
% and at the rare-precursor / DC-flush tumor LN the saturating Hill is the more
% faithful form (Henrickson 2008 PMID 18223650; Moon/Jenkins 2007 PMID 17707129).
% Kept mass-action because it is immaterial in current scenarios — the GVAX helper
% burst is antigen-driven (H_P1_II), not cDC2-count-driven. If harmonized, harmonize
% toward saturation. See notes/model/mechanism_audit_remaining_checklist.md (m1-A20).
%
% MHC-II antigen gate (mid-tier mechanism #3, cA-H4): the antigen-presentation
% Hill on Th priming is H_P1_II (MHC-II), NOT H_P1. H_P1 is the MHC-I
% cross-presentation pMHC pool (cDC1, surface complex + KPR) that licenses CD8;
% conventional Th recognize the tumor antigen on MHC-II via cDC2, a separate
% processing pathway. Th priming used to borrow H_P1, slaving CD4 help to
% cross-presentation efficiency. H_P1_II = V_e.P1/(V_e.P1 + p1_II_50) saturates on
% the same internalized endosomal tumor antigen (cDC2 endocytoses the same
% dying-tumor antigen) but is decoupled from the MHC-I surface-complex/KPR chain;
% the MHC-II loading/display chain is collapsed to QSS (no new species). See
% p1_II_50 in parameters_PDAC.m for provenance (algebraic-derived; scale not yet
% trace-centered, submodel target queued).

% Add Parameters
p = addparameter(model,'k_Th_to_Treg',params.k_Th_to_Treg.Value,'ValueUnits',params.k_Th_to_Treg.Units, 'Notes', [species_name ' differentiation rate to Treg']);
p.UserData.Source = params.k_Th_to_Treg.Notes;
    p.UserData.derivedFrom = params.k_Th_to_Treg.derivedFrom;
p = addparameter(model,'k_CD4_pro',params.k_CD4_pro.Value,'ValueUnits',params.k_CD4_pro.Units, 'Notes', 'CD4 proliferation rate');
p.UserData.Source = params.k_CD4_pro.Notes;
    p.UserData.derivedFrom = params.k_CD4_pro.derivedFrom;
p = addparameter(model,'k_CD4_death',params.k_CD4_death.Value,'ValueUnits',params.k_CD4_death.Units, 'Notes', 'CD4 death rate');
p.UserData.Source = params.k_CD4_death.Notes;
    p.UserData.derivedFrom = params.k_CD4_death.derivedFrom;
K_C2_recruit = addparameter(model,'K_C2_recruit',params.K_C2_recruit.Value,'ValueUnits',params.K_C2_recruit.Units, 'Notes', 'Half-Maximal cancer cell number for T cell recruitment ');
K_C2_recruit.UserData.Source = params.K_C2_recruit.Notes;
    K_C2_recruit.UserData.derivedFrom = params.K_C2_recruit.derivedFrom;

% CD4-help / CD40L licensing of cDC1->CD8 priming (mechanism #7). The H_CD4_help
% multiplier is declared in Teff_module.m (default 1); its governing rule lives
% here because this module owns the V_LN.Th helper pool. Form is an additive
% amplifier with a help-INDEPENDENT floor (mirrors the ArgI #4 fix):
%   H_CD4_help = f_help_indep + (1-f_help_indep)*V_LN.Th/(V_LN.Th+Th_50_help)
% f_help_indep = residual cross-priming licensed by innate DAMP/IL-12 cDC1
% maturation with zero CD4 help (small in PDAC, an innate-poor tumor); the
% Th-driven Hill amplifies above it toward full priming. Th_50_help = half-max
% LN conventional-helper density for licensing (centered on the model's V_LN.Th
% operating range so the Hill is live; submodel anchor queued in
% notes/reconciliation_targets.csv). Uses V_LN.Th (conventional helpers), NOT
% Treg, and not total CD4 -- Treg de-licensing is the separate #8 mechanism.
p = addparameter(model,'f_help_indep',params.f_help_indep.Value,'ValueUnits',params.f_help_indep.Units, 'Notes', 'Help-independent floor fraction of cDC1->CD8 cross-priming (innate DAMP/IL-12 licensing)');
p.UserData.Source = params.f_help_indep.Notes;
    p.UserData.derivedFrom = params.f_help_indep.derivedFrom;
p = addparameter(model,'Th_50_help',params.Th_50_help.Value,'ValueUnits',params.Th_50_help.Units, 'Notes', 'Half-maximal LN conventional-helper (V_LN.Th) density for CD40L licensing of cDC1');
p.UserData.Source = params.Th_50_help.Notes;
    p.UserData.derivedFrom = params.Th_50_help.derivedFrom;
addrule(model,'H_CD4_help = f_help_indep + (1-f_help_indep)*V_LN.Th/(V_LN.Th+Th_50_help)','repeatedAssignment');

% Dedicated division-destiny help-licensing gate, DECOUPLED from the priming-throughput
% H_CD4_help above. Used ONLY in the N_aT costim term (cytokine_tcell_module). Th_50_div
% sits in the realized V_LN.Th band (~1e6) so H_div_help is cold at diagnosis (~0.2) and
% swings under GVAX (~0.75), restoring Path B's costim differential WITHOUT throttling the
% activation reaction's throughput (which keeps the saturated H_CD4_help). No f_help_indep
% floor -- the help-independent baseline divisions live in N_div_base. See cd8_fc doc / the
% double-duty guardrail trace that motivated splitting these two licensing roles.
p = addparameter(model,'Th_50_div',params.Th_50_div.Value,'ValueUnits',params.Th_50_div.Units, 'Notes', 'Half-max V_LN.Th for the division-destiny help-licensing Hill H_div_help (N_aT costim only)');
p.UserData.Source = params.Th_50_div.Notes;
    p.UserData.derivedFrom = params.Th_50_div.derivedFrom;
p = addparameter(model,'H_div_help',1,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Division-destiny help-licensing multiplier (N_aT costim only; decoupled from priming-throughput H_CD4_help)');
addrule(model,'H_div_help = V_LN.Th/(V_LN.Th+Th_50_div)','repeatedAssignment');

% MHC-II antigen-presentation gate for conventional Th priming (mechanism #3).
% H_P1_II saturates on the internalized endosomal tumor antigen (V_e.P1, added by
% antigen_module('1') which runs before this module), decoupled from the MHC-I
% surface-complex/KPR Hill H_P1 that licenses CD8. p1_II_50 is the half-max
% (see header comment + parameters_PDAC.m for provenance).
p = addparameter(model,'p1_II_50',params.p1_II_50.Value,'ValueUnits',params.p1_II_50.Units, 'Notes', 'Half-maximal internalized endosomal tumor-antigen (V_e.P1) concentration for MHC-II presentation gating Th priming');
p.UserData.Source = params.p1_II_50.Notes;
    p.UserData.derivedFrom = params.p1_II_50.derivedFrom;
addparameter(model,'H_P1_II',1,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'MHC-II antigen-presentation Hill gating conventional Th priming (rule below)');
addrule(model,'H_P1_II = V_e.P1/(V_e.P1 + p1_II_50)','repeatedAssignment');

% Add Reactions
% Naive CD4 activation by mature cDC2 (mass-action APC kinetics, Miller 2004 / Marzo 2005).
% Antigen gated by H_P1_II (MHC-II), not H_P1 (MHC-I cross-presentation) — see header comment.
reaction = addreaction(model,'V_LN.nCD4 -> null');
    set(reaction,'ReactionRate','k_T_activation_per_APC*V_LN.mcDC2*H_P1_II*V_LN.nCD4');
    set(reaction,'Notes','Naive CD4 activation by mature cDC2 (MHC-II presentation gate H_P1_II, mechanistic APC kinetics)');
reaction = addreaction(model,'null -> V_LN.aTh');
    set(reaction,'ReactionRate','k_T_activation_per_APC*V_LN.mcDC2*H_P1_II*V_LN.nCD4*n_CD4_clones');
    set(reaction,'Notes','Naive CD4 -> activated Th commitment (x clone diversity). Expansion carried solely by the 2^N_aTh cascade below; the former *n_T_expansion_burst double-counted it (audit V12) and was removed, mirroring CD8.');
% Activated T cell proliferation
reaction = addreaction(model,'V_LN.aTh -> null');
    set(reaction,'ReactionRate','(k_CD4_pro/N_aTh)*V_LN.aTh');
    set(reaction,'Notes',['a' species_name ' cell proliferation']);
reaction = addreaction(model,'null -> V_LN.Th');
    set(reaction,'ReactionRate','(k_CD4_pro/N_aTh)*2^N_aTh*V_LN.aTh');
    set(reaction,'Notes',['a' species_name ' cell proliferation']);
% Differentiation between Treg and Th Cells
reaction = addreaction(model,'V_T.Th -> V_T.Treg');
    set(reaction,'ReactionRate','k_Th_to_Treg*V_T.Th*H_TGFb_Th2Treg');
    set(reaction,'Notes','Differentiation between Treg and Th Cells');

% T cell Death
reaction = addreaction(model,'V_C.Th -> null');
    set(reaction,'ReactionRate','k_CD4_death*V_C.Th');
    set(reaction,'Notes','T cell death in the central compartment');
reaction = addreaction(model,'V_P.Th -> null');
    set(reaction,'ReactionRate','k_CD4_death*V_P.Th');
    set(reaction,'Notes','T cell death in the peripheral compartment');
reaction = addreaction(model,'V_LN.Th -> null');
    set(reaction,'ReactionRate','k_CD4_death*V_LN.Th');
    set(reaction,'Notes','T cell death in the lymph node compartment');
% NOTE (2026-06-22, in-tumor restructure): tumoral Th death is RESTORED, mirroring
% the CD8 fix. It was removed 2026-06-17 to make the cap bind (the drain stack --
% death + TGFb/basal exhaustion + Th->Treg conversion -- nearly cancelled the
% logistic renewal, net ~+0.004/day, so the cap was inert and Th sat ~950 cells far
% below cap). That removal fixed the MEDIAN level but, like CD8, left the no-death
% logistic with a prolif-vs-drain bifurcation that over-disperses cd4_pct (the
% 2026-06-21 bistability diagnosis). Restoring death + relying on the influx floor
% (k_Th_seed*V_T ~1.7e6/day, added AFTER the 2026-06-17 analysis, plus the q_Treg_T_in
% door) puts the level on the same INFLUX/TURNOVER balance as CD8 instead of the
% near-cancellation knife-edge: Th* = influx/(k_CD4_death + Th-drains - k_Th_pro_tumor).
% VERIFIED BY TRACE (not the stale 2026-06-17 collapse arithmetic, which predates the
% seed floor). The data-anchored cap rho_Th_max*V_T is kept as a far safety ceiling.
% Circulating-compartment Th death (V_C/V_P/V_LN above) is unchanged.

% Th exhaustion mechanisms (mirrors CD8 exhaustion routes; previously Th_exh
% was a relabeled death pool, which made Th_exh fraction structurally tied to
% k_CD4_death and crashed CD4 density when forced to match Shibuya 2014).
% Baseline exhaustion from chronic antigen exposure
p = addparameter(model,'k_Th_basal_exh',params.k_Th_basal_exh.Value,'ValueUnits',params.k_Th_basal_exh.Units, 'Notes','Baseline Th exhaustion rate from chronic TCR stimulation (mirrors k_CD8_basal_exh)');
p.UserData.Source = params.k_Th_basal_exh.Notes;
    p.UserData.derivedFrom = params.k_Th_basal_exh.derivedFrom;

% Exhausted-Th (Th_exh) reservoir gateway params (CD4 analog of the CD8 Tpex
% gateway in Teff_module). rho_Thexh_max niche-caps the three Th->Th_exh inflows
% below; k_Th_reflux returns Th_exh -> Th. Cap+reflux together make the Th pool
% mono-stable (rides its rho_Th_max cap) instead of bistable-collapsing. The
% niche-cap ALONE fails (toy ODE: reservoir cannot fill from a collapsed Th).
p = addparameter(model,'rho_Thexh_max',params.rho_Thexh_max.Value,'ValueUnits',params.rho_Thexh_max.Units, 'Notes','Niche carrying-capacity density for the exhausted-Th reservoir (caps Th->Th_exh inflow; mirrors CD8 rho_pex_max)');
p.UserData.Source = params.rho_Thexh_max.Notes;
    p.UserData.derivedFrom = params.rho_Thexh_max.derivedFrom;
p = addparameter(model,'k_Th_reflux',params.k_Th_reflux.Value,'ValueUnits',params.k_Th_reflux.Units, 'Notes','Constitutive exhausted-Th -> functional-Th reflux rate (CD4 analog of k_pex_reflux)');
p.UserData.Source = params.k_Th_reflux.Notes;
    p.UserData.derivedFrom = params.k_Th_reflux.derivedFrom;

reaction = addreaction(model,'V_T.Th -> V_T.Th_exh');
    set(reaction,'ReactionRate','k_Th_basal_exh*V_T.Th*C_total/(C_total+K_C_exh+cell)*(1 - V_T.Th_exh/(rho_Thexh_max*V_T + cell))');
    set(reaction,'Notes','Baseline Th exhaustion from chronic antigen exposure (niche-capped inflow: shuts off as Th_exh reservoir fills, mirrors CD8 gateway)');

% TGFβ-driven Th exhaustion (chronic immunosuppression)
p = addparameter(model,'k_TGFb_Th_exh',params.k_TGFb_Th_exh.Value,'ValueUnits',params.k_TGFb_Th_exh.Units, 'Notes','Maximal Th exhaustion rate driven by TGFβ (chronic immunosuppression)');
p.UserData.Source = params.k_TGFb_Th_exh.Notes;
    p.UserData.derivedFrom = params.k_TGFb_Th_exh.derivedFrom;
reaction = addreaction(model,'V_T.Th -> V_T.Th_exh');
    set(reaction,'ReactionRate','k_TGFb_Th_exh*V_T.Th*H_TGFb_Th_exh*(1 - V_T.Th_exh/(rho_Thexh_max*V_T + cell))');
    set(reaction,'Notes','Th exhaustion driven by TGFβ (cell-specific EC50 via H_TGFb_Th_exh; niche-capped inflow, mirrors CD8 gateway)');

% Th clearance upon antigen clearance (post-tumor-eradication funneling)
reaction = addreaction(model,'V_T.Th -> V_T.Th_exh');
    set(reaction,'ReactionRate','k_cell_clear*V_T.Th*(K_C2_recruit/(C_total^2 + K_C2_recruit))*(1 - V_T.Th_exh/(rho_Thexh_max*V_T + cell))');
    set(reaction,'Notes','Th clearance upon antigen clearance (niche-capped inflow)');

% Constitutive Th_exh -> Th reflux (CD4 analog of CD8 k_pex_reflux). Partial
% reversibility of CD4 exhaustion homeostatically replenishes the functional Th
% pool. With the niche-capped inflows above, this makes Th mono-stable (rides its
% rho_Th_max cap) rather than bistable-collapsing to ~0 (toy ODE 2026-06-18).
% Bounded by the Th renewal cap (rho_Th_max), so it cannot run away.
reaction = addreaction(model,'V_T.Th_exh -> V_T.Th');
    set(reaction,'ReactionRate','k_Th_reflux*V_T.Th_exh');
    set(reaction,'Notes','Constitutive exhausted-Th -> functional-Th reflux (homeostatic replenishment; breaks the Th collapse so rho_Th_max binds)');

% T cell transport
% Central & Peripheral
reaction = addreaction(model,'V_C.Th -> V_P.Th');
    set(reaction,'ReactionRate','q_Treg_P_in*V_C.Th');
    set(reaction,'Notes','T cell transport into the peripheral compartment');
reaction = addreaction(model,'V_P.Th -> V_C.Th');
    set(reaction,'ReactionRate','q_Treg_P_out*V_P.Th');
    set(reaction,'Notes','T cell transport out of the peripheral compartment');

% Central & tumor (with CXCL12-mediated exclusion for PDAC, same as CD8)
reaction = addreaction(model,'V_C.Th -> V_T.Th');
    set(reaction,'ReactionRate','q_Treg_T_in*V_T*V_C.Th*(C_total^2/(C_total^2 + K_C2_recruit))*(1-H_CXCL12_Texcl)*H_vasc');
    set(reaction,'Notes','T cell transport into the tumor compartment (reduced by CXCL12-mediated exclusion; vascular-access gated H_vasc)');
% Central & LN
reaction = addreaction(model,'V_LN.Th -> V_C.Th');
    set(reaction,'ReactionRate','q_Treg_LN_out*V_LN.Th');
    set(reaction,'Notes','T cell transport out of the lymph node compartment');

% ========================================================================
% TUMOR-LOCAL NAIVE CD4 PRIMING (matches ABM t_reg.cuh + dendritic_cell.cuh)
% ========================================================================
% Naive CD4 T cells recruited to tumor encounter mature cDC2s in situ and
% are primed to Th. Downstream TGFb-driven Th->Treg conversion (k_Th_to_Treg
% * H_TGFb_Th2Treg, line 132) handles the Treg fate. The ABM splits naive CD4
% priming 50/50 between Th and Treg at activation time (both k_th and k_treg
% = k_Treg_act), but the QSP's downstream conversion is more mechanistic —
% TGFb-rich regions produce more Treg, TGFb-poor regions produce more Th.
% No H_P1_II gating — tumor-local DCs already matured (requiring antigen).
% Antibody boost on DC maturation is handled by Bcell_module assembly-time
% rewrite on the APC_module maturation reactions.

% Naive CD4 in tumor
addspecies(model.Compartment(3),'nCD4',0,'InitialAmountUnits','cell', 'Notes', 'Naive CD4 T cells in tumor (recruited from blood, primed by local mature cDC2)');

% Trafficking: blood naive CD4 -> tumor (same extravasation as Th/Treg)
reaction = addreaction(model,'V_C.nCD4 -> V_T.nCD4');
    set(reaction,'ReactionRate','q_Treg_T_in*V_T*V_C.nCD4*(C_total^2/(C_total^2 + K_C2_recruit))*(1-H_CXCL12_Texcl)*H_vasc');
    set(reaction,'Notes','Naive CD4 trafficking to tumor (same extravasation gating as Th/Treg)');

% Tumor-local priming: V_T.nCD4 consumed by mature cDC2 -> Th
reaction = addreaction(model,'V_T.nCD4 -> null');
    set(reaction,'ReactionRate','k_T_activation_per_APC*V_T.mcDC2*V_T.nCD4');
    set(reaction,'Notes','Naive CD4 consumed by tumor-local cDC2 priming');
% Production: clonal burst (n_CD4_clones Th per naive consumed)
reaction = addreaction(model,'null -> V_T.Th');
    set(reaction,'ReactionRate','k_T_activation_per_APC*V_T.mcDC2*V_T.nCD4*n_CD4_clones');
    set(reaction,'Notes','Tumor-local CD4 priming by mature cDC2 (clonal burst -> Th; ABM: PARAM_DC_PRIME_K_TH). Treg fate handled by downstream Th->Treg conversion (k_Th_to_Treg*H_TGFb_Th2Treg).');

% Naive CD4 death in tumor (same rate as systemic naive CD4 death)
reaction = addreaction(model,'V_T.nCD4 -> null');
    set(reaction,'ReactionRate','k_nTreg_death*V_T.nCD4');
    set(reaction,'Notes','Naive CD4 death in tumor');
% Antigen-clearance funneling
reaction = addreaction(model,'V_T.nCD4 -> null');
    set(reaction,'ReactionRate','k_cell_clear*V_T.nCD4*(K_C2_recruit/(C_total^2 + K_C2_recruit))');
    set(reaction,'Notes','Naive CD4 clearance upon antigen clearance');

% Tumor-local Th proliferation (PDAC TME-driven)
k_Th_pro_tumor = addparameter(model,'k_Th_pro_tumor',params.k_Th_pro_tumor.Value,...
    'ValueUnits',params.k_Th_pro_tumor.Units,...
    'Notes', 'Tumor-local Th proliferation rate (Ki67+ 20-30% in PDAC TME)');
k_Th_pro_tumor.UserData.Source = params.k_Th_pro_tumor.Notes;
    k_Th_pro_tumor.UserData.derivedFrom = params.k_Th_pro_tumor.derivedFrom;
rho_Th_max = addparameter(model,'rho_Th_max',params.rho_Th_max.Value,...
    'ValueUnits',params.rho_Th_max.Units,...
    'Notes', 'Max Th density in tumor (carrying capacity for local proliferation)');
rho_Th_max.UserData.Source = params.rho_Th_max.Notes;
    rho_Th_max.UserData.derivedFrom = params.rho_Th_max.derivedFrom;
k_Th_seed = addparameter(model,'k_Th_seed',params.k_Th_seed.Value,...
    'ValueUnits',params.k_Th_seed.Units,...
    'Notes', 'Baseline tumor-immunosurveillance CD4 Th seeding density-rate (cancer-gated floor)');
k_Th_seed.UserData.Source = params.k_Th_seed.Notes;
    k_Th_seed.UserData.derivedFrom = params.k_Th_seed.derivedFrom;

% ROUTE-2 FIX 2026-06-22: the CD8-inflammation cap-scaling (alpha_Th_cap +
% CD8_50_Tcap) was RETIRED. The Hill was dead (H_CD8~1 in 94% of posterior
% draws), so it only inflated the data-anchored rho_Th_max cap and injected
% unidentified variance. The Th cap is now the clean rho_Th_max*V_T.
reaction = addreaction(model,'null -> V_T.Th');
    set(reaction,'ReactionRate',...
        'k_Th_pro_tumor*V_T.Th/(1 + V_T.Th/(rho_Th_max*V_T + cell))');
    set(reaction,'Notes','Tumor-local Th renewal, SATURATING in density (rho_Th_max*V_T is the half-saturation density, NOT a hard ceiling). Reformulated 2026-07-13 from the logistic k_pro*Th*(1 - Th/cap) -- same defect as CD8, see notes/model/tcell_carrying_capacity_inversion.md: unclamped, so above the cap the reaction became a quadratic sink scaling with k_Th_pro_tumor, inverting the response. Th sits at ~1.1x the cap at d21 under GVAX (submodel-anchored theta), i.e. right on the sign flip. The saturating form is monotone in k_pro and influx at every density. Bounded by k_CD4_death (restored 2026-06-22), not by the cap. The CD8-inflammation Hill (alpha_Th_cap, route-2) was retired 2026-06-22. k_Th_pro_tumor keeps its Ki67 anchor. Mechanism-specific losses: Th->Treg conversion, Th->Th_exh exhaustion, Th_exh death via k_cell_clear.');
% Tumor-local Th homeostatic death (RESTORED 2026-06-22 in-tumor restructure;
% mirrors CD8). Ordinary CD4 apoptosis, distinct from exhaustion (Th->Th_exh) and
% Th->Treg conversion. Shares the k_CD4_death De Boer & Perelson 1995 anchor with
% the circulating-compartment Th death. Sets the influx/turnover balance level.
reaction = addreaction(model,'V_T.Th -> null');
    set(reaction,'ReactionRate','k_CD4_death*V_T.Th');
    set(reaction,'Notes','Tumor-local homeostatic Th death (apoptosis; restored 2026-06-22). Distinct from exhaustion/conversion; sets the influx/turnover balance level with the trafficking + seed influx.');
% Baseline tumor-immunosurveillance CD4 Th seeding (cancer-gated effector FLOOR; v25, 2026-06-18).
% Direct analog of the v24 CD8 seed (Teff_module.m): the in-tumor Th renewal above is logistic
% (proportional to V_T.Th), so V_T.Th=0 is a fixed point and live Th still bistable-collapses for
% draws where k_Th_pro_tumor falls below its drain stack -- the v20 Th_exh gateway (niche-cap +
% reflux) stabilized the Th_exh pool but did NOT floor LIVE Th, which still craters at the median
% (cd4_helper_density median 0.1 vs obs 17; q95 1671 -> multimodal). The v24 CD8 seed then ADDED
% CD8 to nucleated_total, diluting the already-low cd4_pct further (gvax 9.4->12.1 sigma). This
% restores the missing baseline CD4 TIL immunosurveillance: a minimum Th homing whenever a tumor is
% present, cancer-gated by C_total^2/(C_total^2+K_C2_recruit) (matches the trafficking / Th_exh
% clearance gate, so it switches OFF once antigen is cleared). Seeds LIVE Th, so the Th->Treg /
% Th->Th_exh cascade and the v20 reflux follow naturally. k_Th_seed is an SBI knob (data sets the
% floor). See Teff_module.m k_CD8_seed (v24) and notes/calibration/d21_composition_tension_diagnosis.md.
reaction = addreaction(model,'null -> V_T.Th');
    set(reaction,'ReactionRate','k_Th_seed*V_T*(C_total^2/(C_total^2 + K_C2_recruit))');
    set(reaction,'Notes','Baseline tumor-immunosurveillance CD4 Th seeding (cancer-gated effector floor; analog of k_CD8_seed; prevents unphysical live-Th zero-collapse so cd4_helper_density / cd4_pct stop cratering)');

% Get Model Rules for Updating
model_rules = get(model,'Rules');

% Update Total T Cells in tumor (Rule 3) — includes V_T.nCD4
Tcell_rule = model_rules(3);
rule = get(Tcell_rule,'Rule');
set(Tcell_rule,'Rule',[rule '+V_T.' species_name '+V_T.nCD4']);

% Update Total T Cells in LN (Rule 4)
Tcell_rule = model_rules(4);
rule = get(Tcell_rule,'Rule');
set(Tcell_rule,'Rule',[rule '+V_LN.' species_name]);

warning('off','SimBiology:DimAnalysisNotDone_MatlabFcn_Dimensionless');
