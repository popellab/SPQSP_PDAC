% Enhanced Fibroblast Module (PSC-based) with CAF Subtype Differentiation
%
% Models pancreatic stellate cell (PSC) biology in PDAC tumor microenvironment
% with three functionally distinct CAF subtypes:
%   - iCAF: inflammatory CAFs (IL-6 and CXCL12 secretion -> immunosuppression / T-cell exclusion)
%   - myCAF: myofibroblastic CAFs (ECM, TGFb, VEGF -> desmoplasia)
%   - apCAF: antigen-presenting CAFs (tolerogenic antigen presentation -> Treg induction)
%
% Based on: Elyada 2019 (PMID: 31197017), Biffi 2019 (PMID: 30366930)
%
% KEY FEATURES:
% 1. PSC nomenclature: qPSC (quiescent) -> iCAF/myCAF/apCAF (activated subtypes)
% 2. IL-1-driven iCAF formation (TGFb antagonizes via IL-1R1 downregulation)
% 3. TGFb-driven myCAF formation; IL-6 autocrine feedback promotes iCAF phenotype
% 4. iCAF/myCAF interconversion (phenotypic plasticity)
% 5. Redistributed cytokine production by subtype
% 6. apCAF tolerogenic antigen presentation (Elyada 2019)
% 7. aPSC_total backward-compatibility parameter (sum of all subtypes)
%
% Inputs: model        -- SimBiology model object with four compartments
%         params       -- object containing model parameter Values, Units, and Notes
%
% Outputs: model -- SimBiology model object with enhanced fibroblast module

function model = fibroblast_module_enhanced(model,params)

%% Add Species
qPSC = addspecies(model.Compartment(3),'qPSC',0,'InitialAmountUnits','cell');
    set(qPSC,'Notes',['Quiescent pancreatic stellate cells in tumor compartment. ' ...
        'In normal pancreas, PSCs store vitamin A and maintain stromal homeostasis.']);

iCAF = addspecies(model.Compartment(3),'iCAF',0,'InitialAmountUnits','cell');
    set(iCAF,'Notes',['Inflammatory cancer-associated fibroblasts. ' ...
        'Secrete IL-6 and CXCL12, driving MDSC recruitment, M2 polarization, ' ...
        'and CXCR4-mediated T-cell exclusion (Elyada 2019, Ogawa 2021). ' ...
        'Driven by tumor-derived IL-1alpha; TGFb antagonizes via IL-1R1 repression.']);

myCAF = addspecies(model.Compartment(3),'myCAF',0,'InitialAmountUnits','cell');
    set(myCAF,'Notes',['Myofibroblastic cancer-associated fibroblasts. ' ...
        'Secrete ECM (collagen), TGFb, VEGF. Build desmoplastic barrier. ' ...
        'Favored by TGFb signaling (Biffi 2019). (CXCL12 is iCAF-derived, not myCAF -- audit m2-A-ECM-01.)']);

apCAF = addspecies(model.Compartment(3),'apCAF',0,'InitialAmountUnits','cell');
    set(apCAF,'Notes',['Antigen-presenting cancer-associated fibroblasts. ' ...
        'Express MHC-II, present tumor antigen tolerogenically -> Treg induction. ' ...
        'Rarest subtype ~2-5% of CAFs (Elyada 2019).']);

collagen = addspecies(model.Compartment(3),'collagen',0,'InitialAmountUnits','milligram');
    set(collagen,'Notes',['Fibrillar collagen (types I/III) in tumor (total mass in mg). ' ...
        'Primary structural ECM component driving desmoplasia. ' ...
        'Determines pore size (immune exclusion) and stiffness (mechanotransduction).']);

% PDGF species removed: PDGF-BB is matrix-bound (retention motif) and acts
% as a local paracrine signal, not a freely diffusing cytokine. The Hill
% functions H_PDGF were removed from activation and proliferation reactions
% because PDGF is constitutively present in the PDAC TME at saturating
% levels. See DEVLOG 2026-03-20 for rationale.

CXCL12 = addspecies(model.Compartment(3),'CXCL12',0,'InitialAmountUnits','nanomolarity');
    set(CXCL12,'Notes','CXCL12/SDF-1 chemokine secreted by iCAFs. Major T cell exclusion mechanism in PDAC via CXCR4 signaling (Elyada 2019, Ogawa 2021).');

IL6 = addspecies(model.Compartment(3),'IL6',0,'InitialAmountUnits','nanomolarity');
    set(IL6,'Notes','Interleukin-6 inflammatory cytokine secreted by iCAFs. Promotes MDSC recruitment, M2 polarization, tumor progression.');

IL1 = addspecies(model.Compartment(3),'IL1',0,'InitialAmountUnits','nanomolarity');
    set(IL1,'Notes',['Interleukin-1alpha in tumor compartment. ' ...
        'Tumor-derived; drives iCAF formation via NF-kB/LIF/JAK-STAT3. ' ...
        'TGFb antagonizes signaling by downregulating IL-1R1.']);

%% Add Parameters - PSC Dynamics
k_CAF_encounter = addparameter(model,'k_CAF_encounter',params.k_CAF_encounter.Value,'ValueUnits',params.k_CAF_encounter.Units, 'Notes', 'Rate of encountering resident PSCs as tumor expands into pancreas');
k_CAF_encounter.UserData.Source = params.k_CAF_encounter.Notes;
    k_CAF_encounter.UserData.derivedFrom = params.k_CAF_encounter.derivedFrom;

k_CAF_const = addparameter(model,'k_CAF_const',params.k_CAF_const.Value,'ValueUnits',params.k_CAF_const.Units, 'Notes', 'Constitutive PSC density in pancreatic tissue');
k_CAF_const.UserData.Source = params.k_CAF_const.Notes;
    k_CAF_const.UserData.derivedFrom = params.k_CAF_const.derivedFrom;

% CAF subtype activation rates
p = addparameter(model,'k_PSC_to_myCAF',params.k_PSC_to_myCAF.Value,'ValueUnits',params.k_PSC_to_myCAF.Units, 'Notes', 'qPSC->myCAF activation rate (TGFb-driven)');
p.UserData.Source = params.k_PSC_to_myCAF.Notes;
    p.UserData.derivedFrom = params.k_PSC_to_myCAF.derivedFrom;

p = addparameter(model,'k_PSC_to_iCAF',params.k_PSC_to_iCAF.Value,'ValueUnits',params.k_PSC_to_iCAF.Units, 'Notes', 'qPSC->iCAF activation rate (IL-1-driven)');
p.UserData.Source = params.k_PSC_to_iCAF.Notes;
    p.UserData.derivedFrom = params.k_PSC_to_iCAF.derivedFrom;

p = addparameter(model,'k_PSC_to_apCAF',params.k_PSC_to_apCAF.Value,'ValueUnits',params.k_PSC_to_apCAF.Units, 'Notes', 'qPSC->apCAF activation rate (constitutive low)');
p.UserData.Source = params.k_PSC_to_apCAF.Notes;
    p.UserData.derivedFrom = params.k_PSC_to_apCAF.derivedFrom;

% Interconversion rates
p = addparameter(model,'k_iCAF_to_myCAF',params.k_iCAF_to_myCAF.Value,'ValueUnits',params.k_iCAF_to_myCAF.Units, 'Notes', 'iCAF->myCAF interconversion (TGFb-driven)');
p.UserData.Source = params.k_iCAF_to_myCAF.Notes;
    p.UserData.derivedFrom = params.k_iCAF_to_myCAF.derivedFrom;

p = addparameter(model,'k_myCAF_to_iCAF',params.k_myCAF_to_iCAF.Value,'ValueUnits',params.k_myCAF_to_iCAF.Units, 'Notes', 'myCAF->iCAF interconversion (IL-1-driven)');
p.UserData.Source = params.k_myCAF_to_iCAF.Notes;
    p.UserData.derivedFrom = params.k_myCAF_to_iCAF.derivedFrom;

% Proliferation rates
p = addparameter(model,'k_myCAF_prolif',params.k_myCAF_prolif.Value,'ValueUnits',params.k_myCAF_prolif.Units, 'Notes', 'myCAF proliferation rate');
p.UserData.Source = params.k_myCAF_prolif.Notes;
    p.UserData.derivedFrom = params.k_myCAF_prolif.derivedFrom;

p = addparameter(model,'k_iCAF_prolif',params.k_iCAF_prolif.Value,'ValueUnits',params.k_iCAF_prolif.Units, 'Notes', 'iCAF proliferation rate');
p.UserData.Source = params.k_iCAF_prolif.Notes;
    p.UserData.derivedFrom = params.k_iCAF_prolif.derivedFrom;

p = addparameter(model,'k_apCAF_prolif',params.k_apCAF_prolif.Value,'ValueUnits',params.k_apCAF_prolif.Units, 'Notes', 'apCAF proliferation rate (lower than iCAF/myCAF)');
p.UserData.Source = params.k_apCAF_prolif.Notes;
    p.UserData.derivedFrom = params.k_apCAF_prolif.derivedFrom;

% Stromal carrying capacity (shared across all CAF subtypes)
p = addparameter(model,'f_stroma_max',params.f_stroma_max.Value,'ValueUnits',params.f_stroma_max.Units, 'Notes', 'Max CAF:cancer ratio — total carrying-capacity budget partitioned across iCAF/myCAF/apCAF');
p.UserData.Source = params.f_stroma_max.Notes;
    p.UserData.derivedFrom = params.f_stroma_max.derivedFrom;

% CAF subtype simplex (stick-breaking parameterization).
% Two free parameters (f_apCAF_of_total ∈ (0,1) and
% f_iCAF_of_non_apCAF ∈ (0,1)) parameterize the simplex (apCAF, iCAF,
% myCAF) shares of the total CAF carrying capacity. Subtype caps:
%   apCAF cap  = f_apCAF_of_total * cap_total
%   iCAF cap   = (1 - f_apCAF_of_total) * f_iCAF_of_non_apCAF * cap_total
%   myCAF cap  = (1 - f_apCAF_of_total) * (1 - f_iCAF_of_non_apCAF) * cap_total
% where cap_total = f_stroma_max*C_total + K_stiff_niche*H_stiff_fib.
% Sums to cap_total by construction — preserves the literature-anchored
% total CAF density regardless of how SBI samples the two free params.
% Defaults from Elyada 2019 Fig 5D PDAC scRNA-seq:
%   f_apCAF_of_total = 0.10
%   f_iCAF_of_non_apCAF = 0.222 (so iCAF = 0.9 * 0.222 = 0.20,
%                                 myCAF = 0.9 * 0.778 = 0.70)
p = addparameter(model,'f_apCAF_of_total',params.f_apCAF_of_total.Value,'ValueUnits',params.f_apCAF_of_total.Units, 'Notes', params.f_apCAF_of_total.Notes);
p.UserData.Source = params.f_apCAF_of_total.Notes;
    p.UserData.derivedFrom = params.f_apCAF_of_total.derivedFrom;
p = addparameter(model,'f_iCAF_of_non_apCAF',params.f_iCAF_of_non_apCAF.Value,'ValueUnits',params.f_iCAF_of_non_apCAF.Units, 'Notes', params.f_iCAF_of_non_apCAF.Notes);
p.UserData.Source = params.f_iCAF_of_non_apCAF.Notes;
    p.UserData.derivedFrom = params.f_iCAF_of_non_apCAF.derivedFrom;

% Collagen parameters (qPSC secretion removed as negligible)
p = addparameter(model,'k_col_myCAF_sec',params.k_col_myCAF_sec.Value,'ValueUnits',params.k_col_myCAF_sec.Units, 'Notes', 'Collagen secretion rate by myCAFs');
p.UserData.Source = params.k_col_myCAF_sec.Notes;
    p.UserData.derivedFrom = params.k_col_myCAF_sec.derivedFrom;

p = addparameter(model,'f_col_basal',params.f_col_basal.Value,'ValueUnits','dimensionless', 'Notes', 'Basal fraction of collagen secretion (TGFb-independent)');
p.UserData.Source = params.f_col_basal.Notes;
    p.UserData.derivedFrom = params.f_col_basal.derivedFrom;

p = addparameter(model,'TGFb_50_CAF_act',params.TGFb_50_CAF_act.Value,'ValueUnits',params.TGFb_50_CAF_act.Units, 'Notes', 'TGFb EC50 for CAF activation and interconversion');
p.UserData.Source = params.TGFb_50_CAF_act.Notes;
    p.UserData.derivedFrom = params.TGFb_50_CAF_act.derivedFrom;

p = addparameter(model,'TGFb_50_col_sec',params.TGFb_50_col_sec.Value,'ValueUnits',params.TGFb_50_col_sec.Units, 'Notes', 'TGFb EC50 for collagen secretion');
p.UserData.Source = params.TGFb_50_col_sec.Notes;
    p.UserData.derivedFrom = params.TGFb_50_col_sec.derivedFrom;

p = addparameter(model,'k_col_deg',params.k_col_deg.Value,'ValueUnits',params.k_col_deg.Units, 'Notes', 'Constitutive collagen degradation rate');
p.UserData.Source = params.k_col_deg.Notes;
    p.UserData.derivedFrom = params.k_col_deg.derivedFrom;

p = addparameter(model,'k_MMP_M2',params.k_MMP_M2.Value,'ValueUnits',params.k_MMP_M2.Units, 'Notes', 'M2 macrophage MMP-mediated collagen degradation');
p.UserData.Source = params.k_MMP_M2.Notes;
    p.UserData.derivedFrom = params.k_MMP_M2.derivedFrom;

% phi_cross_50: LOX-crosslinking shield — only nascent (uncrosslinked) collagen is MMP-susceptible.
% f_nascent = phi_cross_50/(phi_cross_50 + phi_collagen). Reconciles fresh-gel in vitro MMP rates
% (Zhu 2001, phi~0 → f_nascent~1) with in vivo fibrotic collagen half-life (phi~0.1-0.5 → f_nascent small).
p = addparameter(model,'phi_cross_50',params.phi_cross_50.Value,'ValueUnits','dimensionless', 'Notes', 'Collagen volume fraction at 50% MMP-susceptibility (LOX-crosslinking shield)');
p.UserData.Source = params.phi_cross_50.Notes;
    p.UserData.derivedFrom = params.phi_cross_50.derivedFrom;
p = addparameter(model,'f_nascent',1,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Fraction of collagen susceptible to M2 MMP degradation (non-crosslinked)');
addrule(model, 'f_nascent = phi_cross_50 / (phi_cross_50 + phi_collagen)', 'repeatedAssignment');

p = addparameter(model,'rho_collagen',params.rho_collagen.Value,'ValueUnits',params.rho_collagen.Units, 'Notes', 'Collagen packing density (material constant)');
p.UserData.Source = params.rho_collagen.Notes;

% phi_collagen: collagen volume fraction (primary ECM state variable).
% Defined against BARE component volumes (not V_T). NOTE: the V_T rule does
% NOT apply a (1 - phi_collagen) packing-saturation factor on cell
% contributions (a factor that earlier text here claimed but that was never
% installed / was removed). An earlier attempt to add that V_T packing factor
% WHILE normalizing phi_collagen against V_T created a circular dependency
% (V_T rule -> phi_collagen -> V_T) and a runaway (phi_collagen amplified as
% V_T shrank under packing saturation, pushing H_pore_*, H_stiff_*, and the
% C_max suppression past their calibration domains). The resolution was to
% normalize phi_collagen against bare volumes (collagen volume / total
% non-saturated volume), which keeps it at its physical value regardless of
% how V_T is structured. Re-introducing V_T packing saturation (audit V9) is
% the first step toward unifying V_T and C_max into one tissue-volume budget;
% it must respect this bare-normalization lesson. See GitHub issue #229
% (V_T<->C_max volume-budget unification) before changing this.
phi_collagen = addparameter(model,'phi_collagen',params.phi_collagen.Value,'ValueUnits','dimensionless','ConstantValue',false, ...
    'Notes', 'Collagen volume fraction (0-1). Bare-volume normalized; primary input for all ECM Hill functions.');
addrule(model, ...
    ['phi_collagen = (V_T.collagen/rho_collagen) / ' ...
     '(V_T.collagen/rho_collagen + ' ...
     '((C_x+C_total)*vol_cell+(CD8_exh+CD8_pex+Th_exh+T_total)*vol_Tcell' ...
     '+M_total*vol_Mcell' ...
     '+V_T.qPSC*vol_qPSCcell+V_T.iCAF*vol_iCAFcell' ...
     '+V_T.myCAF*vol_myCAFcell+V_T.apCAF*vol_apCAFcell)/Ve_T + V_Tmin)'], ...
    'repeatedAssignment');

% Death rates
k_qpsc_death = addparameter(model,'k_qpsc_death',params.k_qpsc_death.Value,'ValueUnits',params.k_qpsc_death.Units, 'Notes', 'Death rate of quiescent PSCs');
k_qpsc_death.UserData.Source = params.k_qpsc_death.Notes;
    k_qpsc_death.UserData.derivedFrom = params.k_qpsc_death.derivedFrom;

p = addparameter(model,'k_iCAF_death',params.k_iCAF_death.Value,'ValueUnits',params.k_iCAF_death.Units, 'Notes', 'Death rate of iCAFs');
p.UserData.Source = params.k_iCAF_death.Notes;
    p.UserData.derivedFrom = params.k_iCAF_death.derivedFrom;

p = addparameter(model,'k_myCAF_death',params.k_myCAF_death.Value,'ValueUnits',params.k_myCAF_death.Units, 'Notes', 'Death rate of myCAFs');
p.UserData.Source = params.k_myCAF_death.Notes;
    p.UserData.derivedFrom = params.k_myCAF_death.derivedFrom;

p = addparameter(model,'k_apCAF_death',params.k_apCAF_death.Value,'ValueUnits',params.k_apCAF_death.Units, 'Notes', 'Death rate of apCAFs');
p.UserData.Source = params.k_apCAF_death.Notes;
    p.UserData.derivedFrom = params.k_apCAF_death.derivedFrom;

%% Pore size — immune cell exclusion (replaces f_ECM_50_T_mot, f_ECM_50_APC_mig)
% d_pore = d_pore_ref * (phi_col_ref / phi_collagen)^0.5
% Exclusion: H_pore = 1 / (1 + (d_pore/d_nuc)^n_pore)

p = addparameter(model,'d_pore_ref',params.d_pore_ref.Value,'ValueUnits',params.d_pore_ref.Units, ...
    'Notes', 'Reference pore size in normal pancreas (Wolf 2013)');
p.UserData.Source = params.d_pore_ref.Notes;

p = addparameter(model,'phi_col_ref',params.phi_col_ref.Value,'ValueUnits',params.phi_col_ref.Units, ...
    'Notes', 'Collagen volume fraction in normal pancreas (reference)');
p.UserData.Source = params.phi_col_ref.Notes;

p = addparameter(model,'d_crit_T',params.d_crit_T.Value,'ValueUnits',params.d_crit_T.Units, ...
    'Notes', 'Critical pore diameter for T cell migration arrest (Wolf 2013)');
p.UserData.Source = params.d_crit_T.Notes;

p = addparameter(model,'d_crit_APC',params.d_crit_APC.Value,'ValueUnits',params.d_crit_APC.Units, ...
    'Notes', 'Critical pore diameter for APC migration arrest (podosome/MMP migration)');
p.UserData.Source = params.d_crit_APC.Notes;

p = addparameter(model,'n_pore',params.n_pore.Value,'ValueUnits',params.n_pore.Units, ...
    'Notes', 'Hill coefficient for pore-size exclusion sharpness');
p.UserData.Source = params.n_pore.Notes;

d_pore = addparameter(model,'d_pore',3.5,'ValueUnits','micrometer','ConstantValue',false, ...
    'Notes', 'Pore size between collagen fibers (decreases with density)');
addrule(model, 'd_pore = d_pore_ref * (phi_col_ref / (phi_collagen + 1e-6))^0.5', 'repeatedAssignment');

H_pore_T = addparameter(model,'H_pore_T',0,'ValueUnits','dimensionless','ConstantValue',false, ...
    'Notes', 'Pore-size-based T cell exclusion. 1=fully excluded, 0=free migration.');
addrule(model, 'H_pore_T = 1 / (1 + (d_pore / d_crit_T)^n_pore)', 'repeatedAssignment');

H_pore_APC = addparameter(model,'H_pore_APC',0,'ValueUnits','dimensionless','ConstantValue',false, ...
    'Notes', 'Pore-size-based APC exclusion. DCs more deformable than T cells.');
addrule(model, 'H_pore_APC = 1 / (1 + (d_pore / d_crit_APC)^n_pore)', 'repeatedAssignment');

%% Stiffness — mechanotransduction (replaces f_ECM_50_T_exh)
% stiffness = E_ref * (phi_collagen / phi_col_ref)^n_stiff
% Exhaustion: H_stiff_Texh = stiffness^n_Texh / (E_50_Texh^n_Texh + stiffness^n_Texh)
% Fibroblast activation: H_stiff_fib = stiffness^n_fib / (E_50_fib^n_fib + stiffness^n_fib)

p = addparameter(model,'E_ref',params.E_ref.Value,'ValueUnits','dimensionless', ...
    'Notes', 'Reference stiffness at phi_col_ref (~1 kPa, dimensionless)');
p.UserData.Source = params.E_ref.Notes;

p = addparameter(model,'n_stiff',params.n_stiff.Value,'ValueUnits','dimensionless', ...
    'Notes', 'Power law exponent for stiffness-collagen relationship');
p.UserData.Source = params.n_stiff.Notes;

p = addparameter(model,'E_50_Texh',params.E_50_Texh.Value,'ValueUnits','dimensionless', ...
    'Notes', 'Half-max stiffness for T cell exhaustion');
p.UserData.Source = params.E_50_Texh.Notes;

p = addparameter(model,'n_Texh',params.n_Texh.Value,'ValueUnits','dimensionless', ...
    'Notes', 'Hill coefficient for stiffness-mediated T cell exhaustion');
p.UserData.Source = params.n_Texh.Notes;

p = addparameter(model,'E_50_fib',params.E_50_fib.Value,'ValueUnits','dimensionless', ...
    'Notes', 'Half-max stiffness for fibroblast mechanotransduction');
p.UserData.Source = params.E_50_fib.Notes;

p = addparameter(model,'n_fib',params.n_fib.Value,'ValueUnits','dimensionless', ...
    'Notes', 'Hill coefficient for stiffness-mediated fibroblast activation');
p.UserData.Source = params.n_fib.Notes;

stiffness = addparameter(model,'stiffness',1.0,'ValueUnits','dimensionless','ConstantValue',false, ...
    'Notes', 'Tissue stiffness (kPa-scaled, dimensionless). Power law of collagen density.');
addrule(model, 'stiffness = E_ref * (phi_collagen / phi_col_ref)^n_stiff', 'repeatedAssignment');

H_stiff_Texh = addparameter(model,'H_stiff_Texh',0,'ValueUnits','dimensionless','ConstantValue',false, ...
    'Notes', 'Stiffness-mediated T cell exhaustion (chronic mechanosensing, PD-L1)');
addrule(model, 'H_stiff_Texh = stiffness^n_Texh / (E_50_Texh^n_Texh + stiffness^n_Texh)', 'repeatedAssignment');

H_stiff_fib = addparameter(model,'H_stiff_fib',0,'ValueUnits','dimensionless','ConstantValue',false, ...
    'Notes', 'Stiffness-mediated fibroblast activation (YAP/TAZ mechanotransduction)');
addrule(model, 'H_stiff_fib = stiffness^n_fib / (E_50_fib^n_fib + stiffness^n_fib)', 'repeatedAssignment');

% phi_col_50_Kmax: carrying capacity ECM barrier
p = addparameter(model,'phi_col_50_Kmax',params.phi_col_50_Kmax.Value,...
    'ValueUnits',params.phi_col_50_Kmax.Units,...
    'Notes',params.phi_col_50_Kmax.Notes);
p.UserData.Source = params.phi_col_50_Kmax.Notes;

% phi_col_max: physical packing ceiling for the logistic collagen-secretion saturation
p = addparameter(model,'phi_col_max',params.phi_col_max.Value,...
    'ValueUnits',params.phi_col_max.Units,...
    'Notes',params.phi_col_max.Notes);
p.UserData.Source = params.phi_col_max.Notes;

% K_stiff_niche: stiffness-dependent CAF niche capacity
p = addparameter(model,'K_stiff_niche',params.K_stiff_niche.Value,...
    'ValueUnits',params.K_stiff_niche.Units,...
    'Notes',params.K_stiff_niche.Notes);
p.UserData.Source = params.K_stiff_niche.Notes;

% Stromal support of carrying capacity
% k_stroma_support: aggregate global multiplier (anchored by Hwang 2008 / Xu 2010
%   submodel targets that constrain total per-CAF carrying-capacity contribution).
% k_stroma_{myCAF,iCAF,apCAF}: subtype-specific weights. myCAFs are the structural
%   /architectural CAF in PDAC (αSMA+, FAP+, collagen producers, peri-tumoral —
%   Elyada 2019, Öhlund 2017 J Exp Med 214:579). iCAFs are inflammatory secretors
%   (IL-6, CXCL12) with minimal structural role; apCAFs are rare MHC-II+
%   immunomodulatory cells (Huang 2022 Cancer Cell). Default ratios
%   myCAF : iCAF : apCAF = 1.0 : 0.2 : 0.05 reflect the structural-vs-inflammatory
%   distinction. LRRC15+ myCAFs specifically correlate with immunotherapy resistance
%   across 6 cancer types (Dominguez 2020 Cancer Discov 10:232).
k_stroma_support = addparameter(model,'k_stroma_support',params.k_stroma_support.Value,...
    'ValueUnits',params.k_stroma_support.Units,'ConstantValue',true,...
    'Notes',params.k_stroma_support.Notes);
k_stroma_support.UserData.Source = params.k_stroma_support.Notes;
k_stroma_support.UserData.derivedFrom = params.k_stroma_support.derivedFrom;

p = addparameter(model,'k_stroma_myCAF',params.k_stroma_myCAF.Value,'ValueUnits',params.k_stroma_myCAF.Units,'Notes',params.k_stroma_myCAF.Notes);
p.UserData.Source = params.k_stroma_myCAF.Notes;
    p.UserData.derivedFrom = params.k_stroma_myCAF.derivedFrom;

p = addparameter(model,'k_stroma_iCAF',params.k_stroma_iCAF.Value,'ValueUnits',params.k_stroma_iCAF.Units,'Notes',params.k_stroma_iCAF.Notes);
p.UserData.Source = params.k_stroma_iCAF.Notes;
    p.UserData.derivedFrom = params.k_stroma_iCAF.derivedFrom;

p = addparameter(model,'k_stroma_apCAF',params.k_stroma_apCAF.Value,'ValueUnits',params.k_stroma_apCAF.Units,'Notes',params.k_stroma_apCAF.Notes);
p.UserData.Source = params.k_stroma_apCAF.Notes;
    p.UserData.derivedFrom = params.k_stroma_apCAF.derivedFrom;

% C_cap0: C-independent baseline carrying-capacity floor (see parameters_PDAC.m).
% Default 0 recovers the prior stroma+vascular-only C_max exactly.
p = addparameter(model,'C_cap0',params.C_cap0.Value,'ValueUnits',params.C_cap0.Units,'ConstantValue',true,'Notes',params.C_cap0.Notes);
p.UserData.Source = params.C_cap0.Notes;
    p.UserData.derivedFrom = params.C_cap0.derivedFrom;

% Restructure C_max: baseline floor + vasculature + subtype-weighted stromal
% support, minus collagen barrier.
% myCAFs dominate the structural carrying capacity; iCAFs/apCAFs contribute little.
% k_stroma_support is the aggregate scale (Hwang 2008 / Xu 2010 submodel anchors).
% C_cap0 is the C-independent floor that gives the small/slow tumor headroom.
cmaxRuleTxt = 'C_max = (C_cap0 + V_T.K_vasc + k_stroma_support * (k_stroma_myCAF*V_T.myCAF + k_stroma_iCAF*V_T.iCAF + k_stroma_apCAF*V_T.apCAF)) * (1 - phi_collagen/(phi_collagen + phi_col_50_Kmax))';
R = sbioselect(model,'Type','rule');
didReplaceCmax = false;
for k = 1:numel(R)
    if contains(strrep(R(k).Rule,' ',''), 'C_max=')
        set(R(k),'Rule', cmaxRuleTxt);
        set(R(k),'Notes','PDAC carrying capacity: vasculature + stromal support - ECM barrier');
        didReplaceCmax = true;
    end
end
if ~didReplaceCmax
    newRule = addrule(model, cmaxRuleTxt, 'repeatedAssignment');
    set(newRule,'Notes','PDAC carrying capacity: vasculature + stromal support - ECM barrier');
end

% Cell volumes
vol_qPSCcell = addparameter(model,'vol_qPSCcell',params.vol_qPSCcell.Value,'ValueUnits',params.vol_qPSCcell.Units, 'Notes', 'Volume of quiescent PSC');
vol_qPSCcell.UserData.Source = params.vol_qPSCcell.Notes;
    vol_qPSCcell.UserData.derivedFrom = params.vol_qPSCcell.derivedFrom;

p = addparameter(model,'vol_iCAFcell',params.vol_iCAFcell.Value,'ValueUnits',params.vol_iCAFcell.Units, 'Notes', 'Volume of iCAF');
p.UserData.Source = params.vol_iCAFcell.Notes;
    p.UserData.derivedFrom = params.vol_iCAFcell.derivedFrom;

p = addparameter(model,'vol_myCAFcell',params.vol_myCAFcell.Value,'ValueUnits',params.vol_myCAFcell.Units, 'Notes', 'Volume of myCAF');
p.UserData.Source = params.vol_myCAFcell.Notes;
    p.UserData.derivedFrom = params.vol_myCAFcell.derivedFrom;

p = addparameter(model,'vol_apCAFcell',params.vol_apCAFcell.Value,'ValueUnits',params.vol_apCAFcell.Units, 'Notes', 'Volume of apCAF');
p.UserData.Source = params.vol_apCAFcell.Notes;
    p.UserData.derivedFrom = params.vol_apCAFcell.derivedFrom;

% PDGF parameters removed (species, secretion, degradation, EC50s).
% See DEVLOG 2026-03-20 for rationale.

% CXCL12 parameters (iCAF-derived, Elyada 2019, Ogawa 2021)
p = addparameter(model,'k_CXCL12_sec_iCAF',params.k_CXCL12_sec_iCAF.Value,'ValueUnits',params.k_CXCL12_sec_iCAF.Units, 'Notes', 'CXCL12 secretion rate by iCAFs');
p.UserData.Source = params.k_CXCL12_sec_iCAF.Notes;
    p.UserData.derivedFrom = params.k_CXCL12_sec_iCAF.derivedFrom;

k_CXCL12_deg = addparameter(model,'k_CXCL12_deg',params.k_CXCL12_deg.Value,'ValueUnits',params.k_CXCL12_deg.Units, 'Notes', 'CXCL12 degradation rate');
k_CXCL12_deg.UserData.Source = params.k_CXCL12_deg.Notes;
    k_CXCL12_deg.UserData.derivedFrom = params.k_CXCL12_deg.derivedFrom;

CXCL12_50_Texcl = addparameter(model,'CXCL12_50_Texcl',params.CXCL12_50_Texcl.Value,'ValueUnits',params.CXCL12_50_Texcl.Units, 'Notes', 'EC50 for CXCL12-mediated T cell exclusion');
CXCL12_50_Texcl.UserData.Source = params.CXCL12_50_Texcl.Notes;
    CXCL12_50_Texcl.UserData.derivedFrom = params.CXCL12_50_Texcl.derivedFrom;

% Effective IL-6 source. C11 collapsed separately fitted iCAF, cancer, and
% TAM source rates into one iCAF-scaled coordinate after their source
% fractions proved nearly invariant across all paper arms.
p = addparameter(model,'k_IL6_sec_iCAF',params.k_IL6_sec_iCAF.Value,'ValueUnits',params.k_IL6_sec_iCAF.Units, 'Notes', 'Effective iCAF-scaled IL-6 source rate');
p.UserData.Source = params.k_IL6_sec_iCAF.Notes;
    p.UserData.derivedFrom = params.k_IL6_sec_iCAF.derivedFrom;

k_IL6_deg = addparameter(model,'k_IL6_deg',params.k_IL6_deg.Value,'ValueUnits',params.k_IL6_deg.Units, 'Notes', 'IL-6 degradation rate');
k_IL6_deg.UserData.Source = params.k_IL6_deg.Notes;
    k_IL6_deg.UserData.derivedFrom = params.k_IL6_deg.derivedFrom;

IL6_50_MDSC = addparameter(model,'IL6_50_MDSC',params.IL6_50_MDSC.Value,'ValueUnits',params.IL6_50_MDSC.Units, 'Notes', 'EC50 for IL-6 enhancement of MDSC recruitment');
IL6_50_MDSC.UserData.Source = params.IL6_50_MDSC.Notes;
    IL6_50_MDSC.UserData.derivedFrom = params.IL6_50_MDSC.derivedFrom;

IL6_50_M2 = addparameter(model,'IL6_50_M2',params.IL6_50_M2.Value,'ValueUnits',params.IL6_50_M2.Units, 'Notes', 'EC50 for IL-6 promotion of M2 polarization');
IL6_50_M2.UserData.Source = params.IL6_50_M2.Notes;
    IL6_50_M2.UserData.derivedFrom = params.IL6_50_M2.derivedFrom;

% IL-6 autocrine feedback on iCAF phenotype (Biffi 2019)
p = addparameter(model,'IL6_50_iCAF',params.IL6_50_iCAF.Value,'ValueUnits',params.IL6_50_iCAF.Units, 'Notes', 'EC50 for IL-6 autocrine feedback promoting iCAF phenotype');
p.UserData.Source = params.IL6_50_iCAF.Notes;
    p.UserData.derivedFrom = params.IL6_50_iCAF.derivedFrom;

p = addparameter(model,'f_IL6_iCAF',params.f_IL6_iCAF.Value,'ValueUnits',params.f_IL6_iCAF.Units, 'Notes', 'Fold-enhancement of iCAF formation by IL-6 feedback');
p.UserData.Source = params.f_IL6_iCAF.Notes;
    p.UserData.derivedFrom = params.f_IL6_iCAF.derivedFrom;

H_IL6_iCAF = addparameter(model,'H_IL6_iCAF',0,'ValueUnits','dimensionless','ConstantValue',false, ...
    'Notes', 'Hill function for IL-6 autocrine feedback on iCAF phenotype');

% IL-1 parameters (tumor-derived, drives iCAF program)
p = addparameter(model,'k_IL1_sec',params.k_IL1_sec.Value,'ValueUnits',params.k_IL1_sec.Units, 'Notes', 'IL-1alpha secretion rate by cancer cells');
p.UserData.Source = params.k_IL1_sec.Notes;
    p.UserData.derivedFrom = params.k_IL1_sec.derivedFrom;

p = addparameter(model,'k_IL1_deg',params.k_IL1_deg.Value,'ValueUnits',params.k_IL1_deg.Units, 'Notes', 'IL-1alpha degradation rate');
p.UserData.Source = params.k_IL1_deg.Notes;
    p.UserData.derivedFrom = params.k_IL1_deg.derivedFrom;

p = addparameter(model,'IL1_50',params.IL1_50.Value,'ValueUnits',params.IL1_50.Units, 'Notes', 'Basal EC50 for IL-1-driven iCAF formation');
p.UserData.Source = params.IL1_50.Notes;
    p.UserData.derivedFrom = params.IL1_50.derivedFrom;

p = addparameter(model,'n_IL1',params.n_IL1.Value,'ValueUnits','dimensionless', ...
    'Notes', 'Hill coefficient for IL-1-driven iCAF formation');
p.UserData.Source = params.n_IL1.Notes;

p = addparameter(model,'TGFb_50_IL1R1',params.TGFb_50_IL1R1.Value,'ValueUnits',params.TGFb_50_IL1R1.Units, 'Notes', 'TGFb EC50 for IL-1R1 downregulation');
p.UserData.Source = params.TGFb_50_IL1R1.Notes;
    p.UserData.derivedFrom = params.TGFb_50_IL1R1.derivedFrom;

p = addparameter(model,'Emax_IL1R1',params.Emax_IL1R1.Value,'ValueUnits','dimensionless', 'Notes', 'Max fold-shift of IL-1 EC50 by TGFb-driven IL-1R1 repression');
p.UserData.Source = params.Emax_IL1R1.Notes;
    p.UserData.derivedFrom = params.Emax_IL1R1.derivedFrom;

% H_IL1_eff: effective IL-1 signaling (TGFb shifts EC50 rightward by reducing IL-1R1)
% IL1_50_eff = IL1_50 * (1 + Emax_IL1R1 * TGFb/(TGFb_50_IL1R1 + TGFb))
% Saturating form: max shift capped at (1 + Emax_IL1R1)-fold even at extreme TGFb
% Use dimensionless ratio form to avoid SimBiology dimensional analysis error with ^n_IL1
IL1_ratio = addparameter(model,'IL1_ratio',0,'ValueUnits','dimensionless','ConstantValue',false, ...
    'Notes', 'Dimensionless IL-1 ratio: IL1 / effective EC50');
addrule(model, 'IL1_ratio = V_T.IL1 / (IL1_50 * (1 + Emax_IL1R1 * V_T.TGFb/(TGFb_50_IL1R1 + V_T.TGFb)))', 'repeatedAssignment');

H_IL1_eff = addparameter(model,'H_IL1_eff',0,'ValueUnits','dimensionless','ConstantValue',false, ...
    'Notes', 'Effective IL-1 Hill function. TGFb antagonizes by increasing effective EC50 via IL-1R1 repression (saturating).');
addrule(model, 'H_IL1_eff = IL1_ratio^n_IL1 / (1 + IL1_ratio^n_IL1)', 'repeatedAssignment');

% Angiogenic factor contribution (myCAF-derived)
p = addparameter(model,'k_vas_myCAF_sec',params.k_vas_myCAF_sec.Value,'ValueUnits',params.k_vas_myCAF_sec.Units, 'Notes', 'Angiogenic factor secretion rate by myCAFs');
p.UserData.Source = params.k_vas_myCAF_sec.Notes;
    p.UserData.derivedFrom = params.k_vas_myCAF_sec.derivedFrom;

% apCAF-specific parameters. The functional coordinate is an effective
% antigen-specific induced-Treg rate; contact geometry is represented by the
% apCAF fraction rather than a second, absolute-cell-count saturation curve.
p = addparameter(model,'k_apCAF_iTreg',params.k_apCAF_iTreg.Value,'ValueUnits',params.k_apCAF_iTreg.Units, 'Notes', 'Effective antigen-specific induced-Treg rate via apCAF presentation');
p.UserData.Source = params.k_apCAF_iTreg.Notes;
    p.UserData.derivedFrom = params.k_apCAF_iTreg.derivedFrom;

p = addparameter(model,'P1_50_apCAF',params.P1_50_apCAF.Value,'ValueUnits',params.P1_50_apCAF.Units, 'Notes', 'EC50 for antigen-dependent Treg induction by apCAFs');
p.UserData.Source = params.P1_50_apCAF.Notes;
    p.UserData.derivedFrom = params.P1_50_apCAF.derivedFrom;

% Backward-compatible aPSC_total parameter (sum of all subtypes)
aPSC_total = addparameter(model,'aPSC_total',0,'ValueUnits','cell','ConstantValue',false, ...
    'Notes', 'Total activated PSC count (iCAF+myCAF+apCAF). Backward compatibility for test statistics.');

% TGFb species, parameters, and base reactions are in cytokine_tcell_module.
% M2 macrophage TGFb secretion is in macrophage_module.

% TGF-β secretion by myCAFs (major source in PDAC)
try
    p = addparameter(model,'k_TGFb_sec_myCAF',params.k_TGFb_sec_myCAF.Value,'ValueUnits',params.k_TGFb_sec_myCAF.Units, 'Notes', 'TGF-β secretion rate by myCAFs');
    p.UserData.Source = params.k_TGFb_sec_myCAF.Notes;
    if isfield(params.k_TGFb_sec_myCAF, 'derivedFrom')
        p.UserData.derivedFrom = params.k_TGFb_sec_myCAF.derivedFrom;
    end
catch
    % Parameter may already exist
end

% TGF-β secretion by myCAFs — bioactive fraction of Bachem 1992 total rate.
% See cytokine_tcell_module notes on f_active_TGFb semantics.
reaction = addreaction(model,'null -> V_T.TGFb');
    set(reaction,'ReactionRate','f_active_TGFb*k_TGFb_sec_myCAF*V_T.myCAF');
    set(reaction,'Notes','Bioactive TGFb secretion by myCAFs (scaled by f_active_TGFb).');

%% Add Reactions - PSC Dynamics

% Encounter resident PSCs as tumor expands into pancreatic tissue
reaction = addreaction(model,'null -> V_T.qPSC');
    set(reaction,'ReactionRate','k_CAF_encounter*V_T*((k_CAF_const-(V_T.qPSC/V_T))/k_CAF_const)');
    set(reaction,'Notes','Encounter resident PSCs as tumor expands into pancreas');

%% Activation: qPSC -> iCAF/myCAF/apCAF
% TGFb drives myCAF; IL-1 (from cancer cells) drives iCAF
% PDGF assumed constitutively saturating in PDAC TME (matrix-bound, see DEVLOG)
% apCAF activation is constitutive (not dependent on TGFb/IL-1)

% qPSC -> myCAF (TGFb-driven, Biffi 2019)
reaction = addreaction(model,'V_T.qPSC -> V_T.myCAF');
    set(reaction,'ReactionRate','k_PSC_to_myCAF*V_T.qPSC*(V_T.TGFb/(TGFb_50_CAF_act + V_T.TGFb))');
    set(reaction,'Notes','PSC activation to myCAF (TGFb-driven, Biffi 2019)');

% qPSC -> iCAF (IL-1-driven, enhanced by IL-6 autocrine feedback)
reaction = addreaction(model,'V_T.qPSC -> V_T.iCAF');
    set(reaction,'ReactionRate','k_PSC_to_iCAF*V_T.qPSC*H_IL1_eff*(1 + f_IL6_iCAF*H_IL6_iCAF)');
    set(reaction,'Notes','PSC activation to iCAF (IL-1-driven + IL-6 feedback)');

% qPSC -> apCAF (constitutive low rate, Elyada 2019)
reaction = addreaction(model,'V_T.qPSC -> V_T.apCAF');
    set(reaction,'ReactionRate','k_PSC_to_apCAF*V_T.qPSC');
    set(reaction,'Notes','PSC activation to apCAF (constitutive low rate, Elyada 2019)');

%% Interconversion (iCAF <-> myCAF plasticity, Biffi 2019)

% iCAF -> myCAF (TGFb-driven)
reaction = addreaction(model,'V_T.iCAF -> V_T.myCAF');
    set(reaction,'ReactionRate','k_iCAF_to_myCAF*V_T.iCAF*(V_T.TGFb/(TGFb_50_CAF_act + V_T.TGFb))');
    set(reaction,'Notes','iCAF->myCAF interconversion (TGFb-driven plasticity)');

% myCAF -> iCAF (IL-1-driven, enhanced by IL-6 autocrine feedback)
reaction = addreaction(model,'V_T.myCAF -> V_T.iCAF');
    set(reaction,'ReactionRate','k_myCAF_to_iCAF*V_T.myCAF*H_IL1_eff*(1 + f_IL6_iCAF*H_IL6_iCAF)');
    set(reaction,'Notes','myCAF->iCAF interconversion (IL-1-driven + IL-6 feedback)');

%% Proliferation (logistic, subtype-partitioned carrying capacity)
% Total stromal carrying-capacity budget: f_stroma_max*C_total + K_stiff_niche*H_stiff_fib
% Each subtype's cap = (share / sum_shares) * total_cap. Normalizing by
% sum_shares makes the partition a simplex — total CAF cap stays at the
% literature-anchored cap_total regardless of how independent share priors
% sample. Without normalization, share-sums < 1 shrink total CAF biomass,
% reducing collagen/stiffness and crashing evolve_to_diagnosis at θ where
% myCAF can't reach the cancer carrying-capacity needed for the 3.85 cm
% target. Logistic saturation acts on each subtype's own count (V_T.<sub>),
% not on aPSC_total — this eliminates the winner-take-all artifact the
% previous shared-cap form produced (more apCAF crowdout myCAF/iCAF,
% ↓collagen, ↑tumor — see #129 / counterfactual sweep 2026-05-01).
% PDGF Hill function removed — proliferation rates represent effective
% in vivo rates at physiological PDGF levels (constitutively saturating
% in PDAC TME).

% myCAF proliferation. Cap = (1 - f_apCAF_of_total) * (1 - f_iCAF_of_non_apCAF) * cap_total
reaction = addreaction(model,'null -> V_T.myCAF');
    set(reaction,'ReactionRate','k_myCAF_prolif*V_T.myCAF*(1 - V_T.myCAF/((1 - f_apCAF_of_total)*(1 - f_iCAF_of_non_apCAF)*(f_stroma_max*C_total + K_stiff_niche*H_stiff_fib) + cell))');
    set(reaction,'Notes','myCAF proliferation: logistic on subtype-partitioned cap (stick-breaking simplex)');

% iCAF proliferation. Cap = (1 - f_apCAF_of_total) * f_iCAF_of_non_apCAF * cap_total
reaction = addreaction(model,'null -> V_T.iCAF');
    set(reaction,'ReactionRate','k_iCAF_prolif*V_T.iCAF*(1 - V_T.iCAF/((1 - f_apCAF_of_total)*f_iCAF_of_non_apCAF*(f_stroma_max*C_total + K_stiff_niche*H_stiff_fib) + cell))');
    set(reaction,'Notes','iCAF proliferation: logistic on subtype-partitioned cap (stick-breaking simplex)');

% apCAF proliferation. Cap = f_apCAF_of_total * cap_total
reaction = addreaction(model,'null -> V_T.apCAF');
    set(reaction,'ReactionRate','k_apCAF_prolif*V_T.apCAF*(1 - V_T.apCAF/(f_apCAF_of_total*(f_stroma_max*C_total + K_stiff_niche*H_stiff_fib) + cell))');
    set(reaction,'Notes','apCAF proliferation: logistic on subtype-partitioned cap (stick-breaking simplex; Elyada 2019)');

%% Death reactions

% qPSC death upon tumor eradication
reaction = addreaction(model,'V_T.qPSC -> null');
    set(reaction,'ReactionRate','k_cell_clear*V_T.qPSC*(K_C2_recruit/(C_total^2 + K_C2_recruit))');
    set(reaction,'Notes','qPSC clearance upon tumor eradication');

% Activated CAF clearance upon tumor eradication
% Modulated by (1-H_stiff_fib): stiff ECM sustains CAFs via mechanotransduction
% niche even when cancer cells are gone (fibrotic scar persistence)
reaction = addreaction(model,'V_T.iCAF -> null');
    set(reaction,'ReactionRate','k_cell_clear*V_T.iCAF*(K_C2_recruit/(C_total^2 + K_C2_recruit))*(1 - H_stiff_fib)');
    set(reaction,'Notes','iCAF clearance upon tumor eradication (suppressed by stiff ECM niche)');

reaction = addreaction(model,'V_T.myCAF -> null');
    set(reaction,'ReactionRate','k_cell_clear*V_T.myCAF*(K_C2_recruit/(C_total^2 + K_C2_recruit))*(1 - H_stiff_fib)');
    set(reaction,'Notes','myCAF clearance upon tumor eradication (suppressed by stiff ECM niche)');

reaction = addreaction(model,'V_T.apCAF -> null');
    set(reaction,'ReactionRate','k_cell_clear*V_T.apCAF*(K_C2_recruit/(C_total^2 + K_C2_recruit))*(1 - H_stiff_fib)');
    set(reaction,'Notes','apCAF clearance upon tumor eradication (suppressed by stiff ECM niche)');

% Constitutive death of qPSC
reaction = addreaction(model,'V_T.qPSC -> null');
    set(reaction,'ReactionRate','k_qpsc_death*V_T.qPSC');
    set(reaction,'Notes','Constitutive death of quiescent PSCs');

% Constitutive death of iCAF
reaction = addreaction(model,'V_T.iCAF -> null');
    set(reaction,'ReactionRate','k_iCAF_death*V_T.iCAF');
    set(reaction,'Notes','Constitutive death of iCAFs');

% Constitutive death of myCAF
reaction = addreaction(model,'V_T.myCAF -> null');
    set(reaction,'ReactionRate','k_myCAF_death*V_T.myCAF');
    set(reaction,'Notes','Constitutive death of myCAFs');

% Constitutive death of apCAF
reaction = addreaction(model,'V_T.apCAF -> null');
    set(reaction,'ReactionRate','k_apCAF_death*V_T.apCAF');
    set(reaction,'Notes','Constitutive death of apCAFs');

%% ECM Dynamics

% Collagen secretion by myCAFs (qPSC secretion removed as negligible <1%)
% Basal fraction f_col_basal is TGFb-independent; remainder is TGFb-driven
% Saturation: logistic (Verhulst) damper (1 - phi_collagen/phi_col_max) →
% secretion shuts off cleanly at the physical packing ceiling phi_col_max,
% structurally bounding phi below the packing limit no matter how large the
% myCAF source grows (audit m2-A-FIB-02). Replaces the (1 - phi_collagen)^2
% quadratic damper, whose only hard ceiling was phi=1 (100% collagen): under
% sustained myCAF expansion (post-therapy fibrotic scar) the quadratic
% attractor drifts toward 1. The logistic form makes the ceiling an explicit,
% literature-anchorable constant (phi_col_max). At the untreated-diagnosis
% attractor (phi well below phi_col_max) the two forms nearly coincide; the
% difference is the post-therapy / high-myCAF scar regime.
reaction = addreaction(model,'null -> V_T.collagen');
    set(reaction,'ReactionRate','k_col_myCAF_sec*(1 - phi_collagen/phi_col_max)*V_T.myCAF*(f_col_basal + (1 - f_col_basal)*V_T.TGFb/(TGFb_50_col_sec + V_T.TGFb))');
    set(reaction,'Notes','Collagen secretion by myCAFs (logistic (1-phi/phi_col_max) cap, f_col_basal + (1-f_col_basal)*H_TGFb_col)');

% Collagen degradation (constitutive + M2 macrophage MMP-dependent)
reaction = addreaction(model,'V_T.collagen -> null');
    set(reaction,'ReactionRate','(k_col_deg + k_MMP_M2 * V_T.Mac_M2 / V_T * f_nascent) * V_T.collagen');
    set(reaction,'Notes','Collagen degradation: constitutive turnover + M2 MMP activity gated by f_nascent (LOX-crosslinking shield)');

% PDGF dynamics removed (species not modeled, see DEVLOG 2026-03-20)

%% CXCL12 Dynamics (T cell exclusion, iCAF-derived per Elyada 2019/Ogawa 2021)

% Cancer-derived CXCL12 parameter
k_CXCL12_sec_cancer = addparameter(model,'k_CXCL12_sec_cancer',params.k_CXCL12_sec_cancer.Value,'ValueUnits',params.k_CXCL12_sec_cancer.Units, 'Notes', 'CXCL12 secretion rate by tumor cells');
k_CXCL12_sec_cancer.UserData.Source = params.k_CXCL12_sec_cancer.Notes;
    k_CXCL12_sec_cancer.UserData.derivedFrom = params.k_CXCL12_sec_cancer.derivedFrom;

% CXCL12 secretion by iCAFs (reassigned from myCAF: scRNA-seq shows CXCL12 is iCAF marker)
reaction = addreaction(model,'null -> V_T.CXCL12');
    set(reaction,'ReactionRate','k_CXCL12_sec_iCAF*V_T.iCAF');
    set(reaction,'Notes','CXCL12 secretion by iCAFs (T cell exclusion, Elyada 2019, Ogawa 2021)');

% CXCL12 secretion by tumor cells
reaction = addreaction(model,'null -> V_T.CXCL12');
    set(reaction,'ReactionRate','k_CXCL12_sec_cancer*C_total');
    set(reaction,'Notes','CXCL12 secretion by PDAC tumor cells (Feig et al. 2013)');

% CXCL12 degradation
reaction = addreaction(model,'V_T.CXCL12 -> null');
    set(reaction,'ReactionRate','k_CXCL12_deg*V_T.CXCL12*V_T');
    set(reaction,'Notes','CXCL12 degradation in tumor');

%% IL-6 Dynamics (one effective iCAF-scaled source)

% C11 reduction (2026-08-24): three source reactions with separately fitted
% per-cell rates were not identifiable from the single bulk IL-6 target. A
% prior-centre trace found stable iCAF/cancer/TAM source fractions across all
% arms and through day 365. Their diagnosis total was therefore absorbed into
% this one iCAF-scaled rate. Cancer- and TAM-derived IL-6 remain acknowledged
% biology, but are not source-attributed in the calibrated paper model.
reaction = addreaction(model,'null -> V_T.IL6');
    set(reaction,'ReactionRate','k_IL6_sec_iCAF*V_T.iCAF');
    set(reaction,'Notes','Effective iCAF-scaled IL-6 source; absorbs the nominal total from retired iCAF/cancer/TAM source attribution (C11)');

% IL-6 degradation
reaction = addreaction(model,'V_T.IL6 -> null');
    set(reaction,'ReactionRate','k_IL6_deg*V_T.IL6*V_T');
    set(reaction,'Notes','IL-6 degradation in tumor');

%% IL-1 Dynamics (tumor-derived, drives iCAF program)

% IL-1alpha secretion by cancer cells
reaction = addreaction(model,'null -> V_T.IL1');
    set(reaction,'ReactionRate','k_IL1_sec*C_total');
    set(reaction,'Notes','IL-1alpha secretion by cancer cells');

% IL-1alpha degradation
reaction = addreaction(model,'V_T.IL1 -> null');
    set(reaction,'ReactionRate','k_IL1_deg*V_T.IL1*V_T');
    set(reaction,'Notes','IL-1alpha degradation in tumor');

%% apCAF Tolerogenic Antigen Presentation (Elyada 2019; Huang 2022)

% apCAFs present tumor antigen to CD4 T cells without canonical costimulation
% and can induce antigen-specific Tregs. Represent that biology as one
% additive contribution to the existing Th -> Treg flux. The apCAF fraction
% captures encounter opportunity without introducing an unidentifiable
% absolute-cell-count gate. P1 is a presentation gate, not an apCAF sink.
if ~isempty(sbioselect(model, 'Type', 'species', 'Name', 'P1')) && ...
   ~isempty(sbioselect(model, 'Type', 'species', 'Name', 'Th'))
    H_apCAF_pMHCII = addparameter(model,'H_apCAF_pMHCII',0,'ValueUnits','dimensionless','ConstantValue',false, ...
        'Notes', 'Quasi-steady antigen-presentation gate for apCAF MHC-II-dependent Treg induction');
    addrule(model, 'H_apCAF_pMHCII = V_T.P1/(V_T.P1 + P1_50_apCAF)', 'repeatedAssignment');

    didAppendApCAF = false;
    for i = 1:length(model.reaction)
        if strcmp(model.reaction(i).reaction, 'V_T.Th -> V_T.Treg')
            baseRate = get(model.reaction(i),'ReactionRate');
            apCAFRate = 'k_apCAF_iTreg*V_T.Th*(V_T.apCAF/(aPSC_total + cell))*H_apCAF_pMHCII';
            set(model.reaction(i),'ReactionRate',['(' baseRate ') + ' apCAFRate]);
            set(model.reaction(i),'Notes','TGFb/ArgI and antigen-specific apCAF contributions to Th->Treg conversion. The Th pool is the model''s local conventional-CD4 substrate; no separate intratumoral naive-CD4 state is represented.');
            didAppendApCAF = true;
            break
        end
    end
    if ~didAppendApCAF
        warning('fibroblast_module_enhanced:missingThToTreg', ...
            'Could not append the apCAF contribution because V_T.Th -> V_T.Treg was not found.');
    end
end

%% Add rules for Hill functions used by immune modules

% CXCL12-mediated T cell exclusion (used by Teff_module)
try
    addrule(model,'H_CXCL12_Texcl = V_T.CXCL12/(CXCL12_50_Texcl + V_T.CXCL12)','repeatedAssignment');
catch
end

% IL-6-mediated MDSC recruitment (used by MDSC_module)
try
    addrule(model,'H_IL6_MDSC = V_T.IL6/(IL6_50_MDSC + V_T.IL6)','repeatedAssignment');
catch
end

% IL-6-mediated M2 polarization (used by macrophage_module)
try
    addrule(model,'H_IL6_M2 = V_T.IL6/(IL6_50_M2 + V_T.IL6)','repeatedAssignment');
catch
end

% IL-6 autocrine feedback on iCAF phenotype
% IL-1 from tumor cells initiates NF-kB -> IL-6 -> JAK/STAT3 -> iCAF program
% IL-1 is explicit (V_T.IL1 species); this captures the downstream autocrine
% loop where iCAF-derived IL-6 reinforces iCAF identity
addrule(model,'H_IL6_iCAF = V_T.IL6/(IL6_50_iCAF + V_T.IL6)','repeatedAssignment');

%% VEGF Contribution (myCAF-derived)

if ~isempty(sbioselect(model, 'Name', 'VEGF'))
    reaction = addreaction(model,'null -> V_T.VEGF');
        set(reaction,'ReactionRate','k_vas_myCAF_sec*V_T.myCAF');
        set(reaction,'Notes','Angiogenic factor secretion by myCAFs (complements M2 macrophage secretion)');
end

%% Rules

% phi_collagen rule is defined above (near parameter declaration)

% aPSC_total backward compatibility (sum of all CAF subtypes)
addrule(model,'aPSC_total = V_T.iCAF + V_T.myCAF + V_T.apCAF','repeatedAssignment');

% --- V_T volume rule: bare-additive (NO (1 - phi_collagen) packing factor) ---
% V_T = V_Tmin + cell_volumes/Ve_T + V_T.collagen/rho_collagen. Cells and
% collagen accumulate independently; collagen never squeezes the cell terms.
%
% This deliberately does NOT apply the (1 - phi_collagen) packing-saturation
% factor on the cell contributions (audit finding V9 / GitHub #229). An earlier
% attempt to install that factor WHILE normalizing phi_collagen against V_T
% created a circular dependency (V_T -> phi_collagen -> V_T) and a runaway
% (phi_collagen amplified as V_T shrank, pushing H_pore_*, H_stiff_*, and the
% C_max collagen barrier past their calibration domains) — see the phi_collagen
% redefinition comment above (~L177) and notes/calibration/
% untreated_tension_analysis_2026-04-28.md. The historical stromal_fraction
% mismatch (pp_median 0.94 vs obs 0.67) that motivated a packing factor was
% instead resolved by (a) the bare-volume phi_collagen normalization and (b) the
% logistic collagen-secretion damper (1 - phi/phi_col_max). stromal_fraction is
% now non-binding (reach_rate ~0.78 in figures/sbi_multi), so V9 is low-priority:
% re-introducing a V_T packing factor is the first step of the V_T<->C_max
% volume-budget unification (#229) and must respect the bare-normalization lesson
% — do not add it in isolation.
%
% Replaces (rather than appends to) the assembled V_T rule from earlier modules
% so the rule is authoritative for all cell types.
model_rules = get(model,'Rules');
volume_rule = model_rules(1);
set(volume_rule, 'Rule', ...
    ['V_T = V_Tmin + ' ...
     '((C_x+C_total)*vol_cell+(CD8_exh+CD8_pex+Th_exh+T_total)*vol_Tcell' ...
     '+M_total*vol_Mcell' ...
     '+V_T.qPSC*vol_qPSCcell+V_T.iCAF*vol_iCAFcell' ...
     '+V_T.myCAF*vol_myCAFcell+V_T.apCAF*vol_apCAFcell)/Ve_T + ' ...
     'V_T.collagen/rho_collagen']);

%% Clean up legacy parameters (if present)
idx = [];
for i = 1:length(model.parameters)
    if strcmp(model.parameters(i).Name, 'k_c')
        idx = [idx i];
    elseif strcmp(model.parameters(i).Name, 'DAMPs')
        idx = [idx i];
    elseif strcmp(model.parameters(i).Name, 'c50')
        idx = [idx i];
    elseif strcmp(model.parameters(i).Name, 'c0')
        idx = [idx i];
    end
end
if ~isempty(idx)
    delete(model.parameters(idx))
end

idx = [];
for i = 1:length(model.species)
    if strcmp(model.species(i).Name, 'c')
        idx = [idx i];
    end
end
if ~isempty(idx)
    delete(model.species(idx))
end

end
