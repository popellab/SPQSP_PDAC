% Macrophage Module
%
% Models macrophage recruitment and polarization in tumor compartment
%
% Inputs: model        -- SimBiology model object with four compartments
%         params       -- object containing model parameter Values, Units, and Notes:
%
% Outputs: model -- SimBiology model object with new macrophage module


function model = macrophage_module(model,params,cancer_types)

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
rho_Mac_max = addparameter(model,'rho_Mac_max',params.rho_Mac_max.Value,'ValueUnits',params.rho_Mac_max.Units, 'Notes', 'Macrophage niche density ceiling (finite stromal niche) ');
rho_Mac_max.UserData.Source = params.rho_Mac_max.Notes;
    rho_Mac_max.UserData.derivedFrom = params.rho_Mac_max.derivedFrom;
k_TGFb_Msec = addparameter(model,'k_TGFb_Msec',params.k_TGFb_Msec.Value,'ValueUnits',params.k_TGFb_Msec.Units, 'Notes', 'Secretion rate of TGFb by macrophage in tumor compartment ');
k_TGFb_Msec.UserData.Source = params.k_TGFb_Msec.Notes;
    k_TGFb_Msec.UserData.derivedFrom = params.k_TGFb_Msec.derivedFrom;
k_vas_Msec = addparameter(model,'k_vas_Msec',params.k_vas_Msec.Value,'ValueUnits',params.k_vas_Msec.Units, 'Notes', 'Secretion rate of angiogenic factor by macrophage in tumor compartment ');
k_vas_Msec.UserData.Source = params.k_vas_Msec.Notes;
    k_vas_Msec.UserData.derivedFrom = params.k_vas_Msec.derivedFrom;
k_IL12_sec = addparameter(model,'k_IL12_sec',params.k_IL12_sec.Value,'ValueUnits',params.k_IL12_sec.Units, 'Notes', 'Secretion rate of IL-12 by mature cDC1 in tumor compartment (cDC1 is the IL-12p70-producing DC subtype; cDC2 do not make IL-12. M1-macrophage IL-12 is a separate source via k_IL12_Msec). Audit m2-A-MAC-01.');
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
p = addparameter(model,'k_IL10_sec_Treg',params.k_IL10_sec_Treg.Value,'ValueUnits',params.k_IL10_sec_Treg.Units, 'Notes', 'Secretion rate of IL-10 by Tregs in tumor compartment');
p.UserData.Source = params.k_IL10_sec_Treg.Notes;
    p.UserData.derivedFrom = params.k_IL10_sec_Treg.derivedFrom;
p = addparameter(model,'k_IL10_sec_MDSC',params.k_IL10_sec_MDSC.Value,'ValueUnits',params.k_IL10_sec_MDSC.Units, 'Notes', 'Secretion rate of IL-10 by MDSCs in tumor compartment');
p.UserData.Source = params.k_IL10_sec_MDSC.Notes;
    p.UserData.derivedFrom = params.k_IL10_sec_MDSC.derivedFrom;
k_IL10_deg = addparameter(model,'k_IL10_deg',params.k_IL10_deg.Value,'ValueUnits',params.k_IL10_deg.Units, 'Notes', 'Degradation rate of IL-10 in tumor compartment ');
k_IL10_deg.UserData.Source = params.k_IL10_deg.Notes;
    k_IL10_deg.UserData.derivedFrom = params.k_IL10_deg.derivedFrom;

k_M2_pol = addparameter(model,'k_M2_pol',params.k_M2_pol.Value,'ValueUnits',params.k_M2_pol.Units, 'Notes', 'Rate of M1 to M2 macrophage polarization ');
k_M2_pol.UserData.Source = params.k_M2_pol.Notes;
    k_M2_pol.UserData.derivedFrom = params.k_M2_pol.derivedFrom;
k_M1_pol = addparameter(model,'k_M1_pol',params.k_M1_pol.Value,'ValueUnits',params.k_M1_pol.Units, 'Notes', 'Rate of M2 to M1 macrophage polarization ');
k_M1_pol.UserData.Source = params.k_M1_pol.Notes;
    k_M1_pol.UserData.derivedFrom = params.k_M1_pol.derivedFrom;
% IL-10 EC50s split per effector context (formerly one IL10_50 used for all three).
% Each Hill is constructed below; priors/anchors are managed per param in pdac_priors.csv
% and calibration_targets/submodel_targets/. See notes/model/pdac_changelog.md (2026-05-23).
IL10_50_M2 = addparameter(model,'IL10_50_M2',params.IL10_50_M2.Value,'ValueUnits',params.IL10_50_M2.Units, 'Notes', 'Half-maximal IL-10 level for M1 to M2 macrophage polarization (innate-myeloid STAT3 axis)');
IL10_50_M2.UserData.Source = params.IL10_50_M2.Notes;
    IL10_50_M2.UserData.derivedFrom = params.IL10_50_M2.derivedFrom;
IL10_50_APC = addparameter(model,'IL10_50_APC',params.IL10_50_APC.Value,'ValueUnits',params.IL10_50_APC.Units, 'Notes', 'Half-maximal IL-10 level for DC maturation suppression (cDC1/cDC2)');
IL10_50_APC.UserData.Source = params.IL10_50_APC.Notes;
    IL10_50_APC.UserData.derivedFrom = params.IL10_50_APC.derivedFrom;
IL10_50_Treg = addparameter(model,'IL10_50_Treg',params.IL10_50_Treg.Value,'ValueUnits',params.IL10_50_Treg.Units, 'Notes', 'Half-maximal IL-10 level for amplifying Treg-mediated CD8 exhaustion');
IL10_50_Treg.UserData.Source = params.IL10_50_Treg.Notes;
    IL10_50_Treg.UserData.derivedFrom = params.IL10_50_Treg.derivedFrom;
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
    % CCL2 secretion by cancer cells (rate ∝ C_total; MDSC contribution not modeled).
    % NOTE: this whole try-block is a FALLBACK — it only runs if the CCL2 species does
    % not already exist. In the assembled PDAC model MDSC_module runs first (script
    % line 90 vs macrophage line 93) and creates CCL2 + this same reaction, so this
    % copy is skipped (addspecies throws -> catch). Annotation kept in sync with the
    % live MDSC_module reaction (audit consensus: "CCL2 cancer-only despite MDSC+cancer
    % annotation" — the live reaction was already correct; this was the stale dupe).
    reaction = addreaction(model,'null -> V_T.CCL2');
        set(reaction,'ReactionRate','k_CCL2_sec*C_total');
        set(reaction,'Notes','CCL2 secretion by cancer cells (C_total); MDSC contribution not modeled');
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
% Recruitment of monocyte-derived macrophages from blood (CCL2/CCR2 axis,
% vascular-access gated H_vasc). Monocytes extravasate and seed the tumor as
% M1-like (classically-activated) macrophages; the M2-like TAM pool is built
% downstream by in-tissue M1->M2 polarization (k_M2_pol*phi_M2_skew, below).
% The earlier formulation also split monocytes M1/M2 AT ENTRY by phi_M2_skew --
% the SAME averaged Hill that drives polarization -- so the M2 drive was applied
% twice and front-loaded the M2/M1 ratio (v26 d21 trace: 11.5 vs obs ~1.4).
% Direct monocyte->M2 differentiation in a suppressive milieu (Movahedi 2010
% PMID 20424128, Franklin 2014 PMID 24812208) is retained as fast POST-ENTRY
% polarization rather than a separate entry channel, so phi_M2_skew governs the
% split through one pathway. Recruitment is gated by a density-dependent niche
% carrying cap (1 - M_total/(rho_Mac_max*V_T)): TAMs saturate a finite stromal
% niche rather than accumulating unboundedly (v26 total TAM ~3.2x high, no cap).
% Total TAM still tends to R/k_Mac_death at the ceiling, preserving the Ren-2017
% pan-macrophage k_Mac_rec anchor. See
% notes/calibration/d21_composition_tension_diagnosis.md (macrophage trace+toy).
reaction = addreaction(model,'null -> V_T.Mac_M1');
    set(reaction,'ReactionRate','k_Mac_rec*V_T*(V_T.CCL2/(V_T.CCL2 + CCL2_50))*H_vasc*(1 - (V_T.Mac_M1 + V_T.Mac_M2)/(rho_Mac_max*V_T + cell))');
    set(reaction,'Notes','Recruitment of CCL2-attracted monocytes seeding as M1 (vascular-access gated H_vasc; density-capped at the rho_Mac_max niche ceiling). M2-like TAMs form downstream via M1->M2 polarization.');
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

% Secretion of IL-10 by M2 macrophages
reaction = addreaction(model,'null -> V_T.IL10');
    set(reaction,'ReactionRate','k_IL10_sec*V_T.Mac_M2');
    set(reaction,'Notes','Secretion of IL-10 by macrophage in tumor');

% Arginase-I secretion by M2 TAMs (mechanism #7, cA-M3): the DIRECT M2->CD8
% suppression edge. M2/alternatively-activated TAMs are Arg1-high (the defining
% M2 metabolic program), so they feed the SAME V_T.ArgI pool created in
% MDSC_module (which runs before this module). The existing arginase chain then
% carries the M2 contribution to the killing rate with no new gate or EC50:
% H_ArgI_Teff = V_T.ArgI/(ArgI_50_Teff+V_T.ArgI) -> H_MDSC = 1-(1-H_NO)*(1-H_ArgI_Teff)
% -> (1-H_MDSC) on k_C_Tcell_eff. Before this, M2 suppressed CD8
% only INDIRECTLY via TGFb (line 187) and IL-10 (line 192). Per-cell arginase rate
% reused from MDSCs (k_ArgI_sec, Rodriguez PMID 19201693) -- both are Arg1-high
% myeloid cells and the aggregate V_T.ArgI is unidentifiable by source. NO is NOT
% added here: iNOS/NO is the M1 (classically-activated) product, not M2.
reaction = addreaction(model,'null -> V_T.ArgI');
    set(reaction,'ReactionRate','k_ArgI_sec*V_T.Mac_M2');
    set(reaction,'Notes','Arginase-I secretion by M2 TAMs (direct CD8 suppression via the shared V_T.ArgI pool; mechanism #7 cA-M3). Per-cell rate reused from MDSC k_ArgI_sec.');

% Secretion of IL-10 by Tregs (Pan 2025 Front Immunol DOI:10.3389/fimmu.2025.1582305
% identifies Tregs as IL-10 producers in PDAC TME; canonical Treg immunosuppression
% mechanism). Distributing IL-10 across multiple producers decouples [IL-10] from
% being a direct proxy for [M2] — addresses PDAC_IL10 ↔ m2tam joint-reach failure.
reaction = addreaction(model,'null -> V_T.IL10');
    set(reaction,'ReactionRate','k_IL10_sec_Treg*V_T.Treg');
    set(reaction,'Notes','IL-10 secretion by Tregs (Pan 2025 DOI:10.3389/fimmu.2025.1582305)');

% Secretion of IL-10 by MDSCs (Pan 2025: "MDSCs secrete IL-10 and TGF-β to drive
% Treg differentiation"). Co-secreted with TGFβ as part of the MDSC suppression
% program; provides a second cancer-mass-coupled IL-10 source.
reaction = addreaction(model,'null -> V_T.IL10');
    set(reaction,'ReactionRate','k_IL10_sec_MDSC*V_T.MDSC');
    set(reaction,'Notes','IL-10 secretion by MDSCs (Pan 2025 DOI:10.3389/fimmu.2025.1582305)');

% Degradation of IL-10
reaction = addreaction(model,'V_T.IL10 -> null');
    set(reaction,'ReactionRate','k_IL10_deg*V_T.IL10');
    set(reaction,'Notes','Degradation of IL-10 in tumor');

% M1 to M2 polarization (driven by TGF-β, IL-10, and IL-6 via phi_M2_skew)
% phi_M2_skew is the averaged M2-skewing milieu drive
%   (H_TGFb_M2pol + H_IL10_M2 + H_IL6_M2)/3   [TGFb Hill inlined via TGFb_50_M2pol]
% (declared with the Hill rules below). Averaged Hill logic replaces the earlier
% OR-gate 1 - (1-H_TGFb)(1-H_IL10)(1-H_IL6_M2), which numerically saturates to 1.0
% as soon as any two of the three Hills sit near their EC50 plateau — producing a
% "constitutively-maximal" polarization rate in PDAC where TGFb, IL-10, and IL-6 are
% all chronically above their bioactive EC50s. Averaging preserves the monotone
% structure (-> 1 on saturation, -> 0 when all axes absent), lets each axis contribute
% a 1/3 fractional lever, and matches the multi-subtype biology where IL-10, TGFb, and
% IL-6 drive distinct M2 programs that sum rather than short-circuit. The SAME
% phi_M2_skew also sets the direct monocyte->M2 recruitment fraction above, so the
% entry-fate split and the polarization drive are locked to one expression.
reaction = addreaction(model,'V_T.Mac_M1 -> V_T.Mac_M2');
    set(reaction,'ReactionRate','k_M2_pol*V_T.Mac_M1*phi_M2_skew');
    set(reaction,'Notes','M1 to M2 macrophage polarization (averaged Hill drive phi_M2_skew over TGFb, IL-10, IL-6 — max rate at k_M2_pol when all three are saturated).');
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

model = phagocytosis_module(model,params);

IL10_50_phago = addparameter(model,'IL10_50_phago',params.IL10_50_phago.Value,'ValueUnits',params.IL10_50_phago.Units, 'Notes', 'Half-maximal IL-10 level for phagocytosis by macrophage ');
IL10_50_phago.UserData.Source = params.IL10_50_phago.Notes;
    IL10_50_phago.UserData.derivedFrom = params.IL10_50_phago.derivedFrom;
K_M1_phago = addparameter(model,'K_M1_phago',params.K_M1_phago.Value,'ValueUnits',params.K_M1_phago.Units, 'Notes', 'Association coefficient for macrophage and cancer cell ');
K_M1_phago.UserData.Source = params.K_M1_phago.Notes;
    K_M1_phago.UserData.derivedFrom = params.K_M1_phago.derivedFrom;

M_total = addparameter(model,'M_total',0,'ValueUnits','cell','ConstantValue',false, 'Notes', 'Total number of macrophages ');
% IL-10 Hills split per effector context (see IL10_50_M2/APC/Treg above).
% (M1->M2 IL-10 drive is inlined in the polarization reaction rate, not via a named Hill.)
H_IL10_APC = addparameter(model,'H_IL10_APC',1,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill: IL-10 suppression of DC maturation (cDC1/cDC2)');
H_IL10_Treg = addparameter(model,'H_IL10_Treg',1,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill: IL-10 amplification of Treg-mediated CD8 exhaustion');
H_IL10_phago = addparameter(model,'H_IL10_phago',1,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function for IL10-mediated inhibition on phagocytosis ');
H_IL12 = addparameter(model,'H_IL12',1,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function for IL12 ');

% IL-6-mediated enhancement of M2 polarization (PDAC-specific)
% Create Hill function (will be 0 unless rule is added by fibroblast module)
H_IL6_M2 = addparameter(model,'H_IL6_M2',0,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function for IL-6 promotion of M2 macrophage polarization');
% Rule will be added by fibroblast_module_enhanced if IL-6 exists

% Averaged M2-skewing milieu drive, shared by (i) the direct monocyte->M2
% recruitment split and (ii) the M1->M2 polarization reaction so the two can
% never drift apart. Bounded 0 (no M2 drive) -> 1 (TGFb/IL-10/IL-6 all saturated).
phi_M2_skew = addparameter(model,'phi_M2_skew',0,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Averaged M2-polarizing milieu drive (TGFb/IL-10/IL-6); sets both the fraction of recruited monocytes differentiating directly to M2 and the M1->M2 polarization drive');

addrule(model,'M_total = V_T.Mac_M1+V_T.Mac_M2','repeatedAssignment');
addrule(model,'H_IL10_APC = V_T.IL10/(IL10_50_APC+V_T.IL10)','repeatedAssignment');
addrule(model,'H_IL10_Treg = V_T.IL10/(IL10_50_Treg+V_T.IL10)','repeatedAssignment');
addrule(model,'H_IL10_phago = V_T.IL10/(IL10_50_phago+V_T.IL10)','repeatedAssignment');
addrule(model,'H_IL12 = V_T.IL12/(IL12_50+V_T.IL12)','repeatedAssignment');
addrule(model,'phi_M2_skew = (V_T.TGFb/(V_T.TGFb+TGFb_50_M2pol) + V_T.IL10/(V_T.IL10+IL10_50_M2) + H_IL6_M2)/3','repeatedAssignment');

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

% IL-10 gating of the Treg-mediated CD8 exhaustion increment.
% This multiplicative '*H_IL10_Treg' makes the Treg-driven CD8->CD8_pex term
% (k_Treg_CD8_exh*CD8*Tregs_/(...), created in Teff_module.m) cytokine-dependent.
% NOTE (2026-06-17): the Tpex keystone retargeted the four exhaustion routes from
% '-> V_T.CD8_exh' to the gateway '-> V_T.CD8_pex'. The string match below must
% track that target or the *H_IL10_Treg gate silently drops off (it did, between
% commits f843bf6 and this fix). Guarded by the k_Treg_CD8_exh rate-string so only
% the Treg route is gated, not the other three pex-gateway routes.
% The mechanism-audit consensus flagged this as a "required gate that should be an
% additive amplifier" (the #4 ArgI-sibling concern). ADJUDICATED FALSE-POSITIVE
% (2026-06-04, kept multiplicative on purpose) for two reasons:
%   (1) This is one of FIVE CD8->CD8_exh routes in Teff_module.m (234-253); the
%       other four (chronic-antigen k_CD8_basal_exh, antigen-clearance k_cell_clear,
%       cancer PD-1 k_C_CD8_exh, ECM stiffness k_stiff_CD8_exh) are IL-10-INDEPENDENT.
%       So the model does NOT make CD8 exhaustion IL-10-dependent overall -- only the
%       Treg-specific increment. The large IL-10-independent floor lives in those four.
%   (2) k_Treg_CD8_exh is submodel-anchored to Sawant 2019 Fig 3d, where the forward
%       model defines it as the increment at SATURATING IL-10/IL-35 (WT, H_IL10_Treg->1).
%       The Treg-specific IL-10/IL-35 double-KO arm collapses CD8 exhaustion 52.6%->0.1%
%       (~basal) WHILE CTLA-4 / IL-2-deprivation / contact machinery remain intact ->
%       it directly measures the IL-10-independent Treg exhaustion floor as ~0. An
%       additive (a + (1-a)*H_IL10_Treg) form would just recover a~0, i.e. this gate.
% Contrast with #4 (MDSC_module ArgI): there the anchor (Davidson, arginase-FREE)
% CONTRADICTED the gate -> real bug, made additive. Here the anchor VALIDATES it.
% Residual simplification (not a sign error): the model has no IL-35 species, so
% H_IL10_Treg is a lumped IL-10/IL-35 Treg-cytokine proxy; IL10_50_Treg's EC50 is
% IL-10-specific (Gorby 2020) while the requirement is anchored to the joint KO.
for i = 1:length(model.reaction)
    if strcmp(model.reaction(i).reaction, 'V_T.CD8 -> V_T.CD8_pex') && ~isempty(strfind(model.reaction(i).ReactionRate, 'k_Treg_CD8_exh'))
        model.reaction(i).ReactionRate = [model.reaction(i).ReactionRate, '*H_IL10_Treg'];
    end
end
