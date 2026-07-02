% Enhanced Fibroblast Module (PSC-based) with CAF Subtype Differentiation
%
% Models pancreatic stellate cell (PSC) biology in PDAC tumor microenvironment
% with three functionally distinct CAF subtypes:
%   - iCAF: inflammatory CAFs (IL-6, CCL5 secretion -> immunosuppression)
%   - myCAF: myofibroblastic CAFs (ECM, CXCL12, TGFb, VEGF -> desmoplasia)
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
        'Secrete IL-6 and CCL5, driving MDSC recruitment and M2 polarization. ' ...
        'Driven by tumor-derived IL-1alpha; TGFb antagonizes via IL-1R1 repression.']);

myCAF = addspecies(model.Compartment(3),'myCAF',0,'InitialAmountUnits','cell');
    set(myCAF,'Notes',['Myofibroblastic cancer-associated fibroblasts. ' ...
        'Secrete ECM, CXCL12, TGFb, VEGF. Build desmoplastic barrier. ' ...
        'Favored by TGFb signaling (Biffi 2019).']);

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
% Defined against BARE component volumes (not V_T) so it stays stable when
% the V_T rule applies the (1 - phi_collagen) packing-saturation factor on
% cell contributions. Using V_T in the denominator created a circular
% dependency once the V_T rule referenced phi_collagen, and also amplified
% phi_collagen as V_T shrank under packing saturation — pushing all
% downstream Hill functions (H_pore_*, H_stiff_*, C_max suppression, etc.)
% past their original calibration domains. Computing against bare volumes
% keeps phi_collagen at its physical value (collagen volume / total
% non-saturated volume) regardless of how the V_T rule is structured.
phi_collagen = addparameter(model,'phi_collagen',params.phi_collagen.Value,'ValueUnits','dimensionless','ConstantValue',false, ...
    'Notes', 'Collagen volume fraction (0-1). Bare-volume normalized; primary input for all ECM Hill functions.');
addrule(model, ...
    ['phi_collagen = (V_T.collagen/rho_collagen) / ' ...
     '(V_T.collagen/rho_collagen + ' ...
     '((C_x+C_total)*vol_cell+(CD8_exh+Th_exh+T_total)*vol_Tcell' ...
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

% K_stiff_niche: stiffness-dependent CAF niche capacity
p = addparameter(model,'K_stiff_niche',params.K_stiff_niche.Value,...
    'ValueUnits',params.K_stiff_niche.Units,...
    'Notes',params.K_stiff_niche.Notes);
p.UserData.Source = params.K_stiff_niche.Notes;

% Stromal support of carrying capacity
k_stroma_support = addparameter(model,'k_stroma_support',params.k_stroma_support.Value,...
    'ValueUnits',params.k_stroma_support.Units,'ConstantValue',true,...
    'Notes',params.k_stroma_support.Notes);
k_stroma_support.UserData.Source = params.k_stroma_support.Notes;
k_stroma_support.UserData.derivedFrom = params.k_stroma_support.derivedFrom;

% Restructure C_max: vasculature + stromal support, minus collagen barrier
% CAFs provide nutrients/growth factors that partially compensate for poor vasculature
% Collagen barrier is intentionally weak — stroma depletion worsens PDAC (Rhim 2014, Ozdemir 2014)
cmaxRuleTxt = 'C_max = (V_T.K + k_stroma_support * aPSC_total) * (1 - phi_collagen/(phi_collagen + phi_col_50_Kmax))';
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

% IL-6 parameters (iCAF-derived)
p = addparameter(model,'k_IL6_sec_iCAF',params.k_IL6_sec_iCAF.Value,'ValueUnits',params.k_IL6_sec_iCAF.Units, 'Notes', 'IL-6 secretion rate by iCAFs');
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

% apCAF-specific parameters
p = addparameter(model,'k_P1_up_apCAF',params.k_P1_up_apCAF.Value,'ValueUnits',params.k_P1_up_apCAF.Units, 'Notes', 'Antigen uptake rate by apCAFs');
p.UserData.Source = params.k_P1_up_apCAF.Notes;
    p.UserData.derivedFrom = params.k_P1_up_apCAF.derivedFrom;

p = addparameter(model,'k_apCAF_Treg',params.k_apCAF_Treg.Value,'ValueUnits',params.k_apCAF_Treg.Units, 'Notes', 'Max Th->Treg conversion rate via apCAF');
p.UserData.Source = params.k_apCAF_Treg.Notes;
    p.UserData.derivedFrom = params.k_apCAF_Treg.derivedFrom;

p = addparameter(model,'K_apCAF_Treg',params.K_apCAF_Treg.Value,'ValueUnits',params.K_apCAF_Treg.Units, 'Notes', 'Half-max apCAF count for Treg induction saturation');
p.UserData.Source = params.K_apCAF_Treg.Notes;
    p.UserData.derivedFrom = params.K_apCAF_Treg.derivedFrom;

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
% Saturation: (1 - phi_collagen)^2 → quadratic damper for sharper cap as
% collagen fills volume. Linear (1-phi) damping was too gentle: secretion
% at phi=0.78 was 22% of unsaturated rate, allowing collagen to accumulate
% to a phi_col attractor at 0.78 (notes/calibration/untreated_tension_analysis_2026-04-28.md).
% Quadratic form drops secretion to 5% at phi=0.78 and 25% at phi=0.5,
% targeting a more physiological phi_col attractor in the 0.4-0.5 range.
reaction = addreaction(model,'null -> V_T.collagen');
    set(reaction,'ReactionRate','k_col_myCAF_sec*(1 - phi_collagen)^2*V_T.myCAF*(f_col_basal + (1 - f_col_basal)*V_T.TGFb/(TGFb_50_col_sec + V_T.TGFb))');
    set(reaction,'Notes','Collagen secretion by myCAFs ((1-phi)^2 cap, f_col_basal + (1-f_col_basal)*H_TGFb_col)');

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

%% IL-6 Dynamics (inflammatory/immunosuppressive, iCAF-derived)

% IL-6 secretion by iCAFs
reaction = addreaction(model,'null -> V_T.IL6');
    set(reaction,'ReactionRate','k_IL6_sec_iCAF*V_T.iCAF');
    set(reaction,'Notes','IL-6 secretion by iCAFs (drives MDSC recruitment and M2 polarization)');

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

%% CCL5 Secretion by iCAFs (moved from CCL5_Treg_module)
% CCL5 is inflammatory -> attributed to iCAFs

% CCL5 secretion parameter (iCAF-derived)
p = addparameter(model,'k_CCL5_sec_iCAF',params.k_CCL5_sec_iCAF.Value,'ValueUnits',params.k_CCL5_sec_iCAF.Units, 'Notes', 'CCL5 secretion rate by iCAFs');
p.UserData.Source = params.k_CCL5_sec_iCAF.Notes;
    p.UserData.derivedFrom = params.k_CCL5_sec_iCAF.derivedFrom;

% Note: CCL5 species is created by CCL5_Treg_module. This reaction is added
% after CCL5 species exists because fibroblast_module runs after CCL5_Treg_module
% in assembly. If CCL5 species doesn't exist yet, skip gracefully.
if ~isempty(sbioselect(model, 'Type', 'species', 'Name', 'CCL5'))
    reaction = addreaction(model,'null -> V_T.CCL5');
        set(reaction,'ReactionRate','k_CCL5_sec_iCAF*V_T.iCAF');
        set(reaction,'Notes','CCL5 secretion by iCAFs (inflammatory chemokine, Treg recruitment)');
end

%% apCAF Tolerogenic Antigen Presentation (Elyada 2019)

% Reaction A: Antigen sequestration by apCAFs
% apCAFs take up P1 but do NOT process it into productive T cell activation - it's a sink
% Competes with professional APC uptake (k_up * V_T.APC * V_T.P1 * V_T)
if ~isempty(sbioselect(model, 'Type', 'species', 'Name', 'P1'))
    reaction = addreaction(model,'V_T.P1 -> null');
        set(reaction,'ReactionRate','k_P1_up_apCAF*V_T.apCAF*V_T.P1*V_T');
        set(reaction,'Notes','Antigen sequestration by apCAFs (competes with professional APC uptake, Elyada 2019)');
end

% Reaction B: Tolerogenic Th -> Treg conversion by apCAFs
% ADDITIONAL to existing TGFb-dependent Th->Treg pathway in Th_module
% Creates distinct antigen-dependent, cell-contact-dependent tolerance circuit
% H_P1_apCAF rule for antigen-dependent Treg induction
if ~isempty(sbioselect(model, 'Type', 'species', 'Name', 'P1')) && ...
   ~isempty(sbioselect(model, 'Type', 'species', 'Name', 'Th'))
    H_P1_apCAF = addparameter(model,'H_P1_apCAF',0,'ValueUnits','dimensionless','ConstantValue',false, ...
        'Notes', 'Hill function for antigen-dependent apCAF Treg induction');
    addrule(model, 'H_P1_apCAF = V_T.P1/(V_T.P1 + P1_50_apCAF)', 'repeatedAssignment');

    reaction = addreaction(model,'V_T.Th -> V_T.Treg');
        set(reaction,'ReactionRate','k_apCAF_Treg*V_T.Th*(V_T.apCAF/(V_T.apCAF + K_apCAF_Treg))*H_P1_apCAF');
        set(reaction,'Notes','Tolerogenic Th->Treg conversion by apCAFs (antigen-dependent, Elyada 2019)');
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

% --- Option A: packing-saturation on cell volume contributions to V_T ---
% Without saturation, cells and collagen accumulated independently in V_T:
% V_T = V_Tmin + cell_volumes/Ve_T + V_T.collagen/rho_collagen.
% That additive structure produced nonphysical states where the model fit
% observed collagen mass but mismatched stromal_fraction (pp_median 0.94
% vs obs 0.67) — see notes/calibration/untreated_tension_analysis_2026-04-28.md.
% The (1 - phi_collagen) factor on cell terms models cells being squeezed
% by collagen as it fills the tumor volume. phi_collagen is now defined
% against bare component volumes (see redefinition above), so V_T's
% repeatedAssignment rule has no circular dependency, and the bare-volume
% phi_collagen also stays stable when V_T shrinks — keeping all downstream
% Hill functions (H_pore_*, H_stiff_*, C_max suppression) at their
% calibration domain.
%
% Replaces (rather than appends to) the assembled V_T rule from earlier
% modules so the saturation factor applies to all cell types uniformly.
model_rules = get(model,'Rules');
volume_rule = model_rules(1);
set(volume_rule, 'Rule', ...
    ['V_T = V_Tmin + ' ...
     '((C_x+C_total)*vol_cell+(CD8_exh+Th_exh+T_total)*vol_Tcell' ...
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
