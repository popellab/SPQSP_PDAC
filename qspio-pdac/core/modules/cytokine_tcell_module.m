% T-cell Cytokine Module
%
% Centralizes all T-cell related cytokine species, parameters, reactions,
% and activation generation rules.
%
% Assembly position: After Th_module, before MDSC_module.
% All T-cell species (nCD8, aCD8, CD8, nCD4, aTreg, Treg, aTh, Th) exist.
%
% Reactions are split into two sections:
%
%   CLONE-SPECIFIC — Processes performed only by the antigen-specific clone
%     tracked in the model (e.g., IL2/IFNg secretion upon TCR activation).
%
%   SHARED — Consumption, degradation, and constitutive secretion reactions
%     centralized from the individual T-cell modules. Rate expressions match
%     the original implementations exactly (no diversity scaling).
%
% NOTE: Diversity scaling (×div_CD8/div_Treg) on shared reactions is a
% candidate structural change for a future commit.
%
% Inputs: model  -- SimBiology model object with T-cell species already added
%         params -- object containing model parameter Values, Units, and Notes
%
% Outputs: model -- SimBiology model with cytokine reactions added

function model = cytokine_tcell_module(model, params)

%% Species
% IL-2 is no longer a bulk dynamic species. It is represented as a LOCAL niche
% concentration (IL2_niche) computed at quasi-steady-state from the priming-cluster
% secretion/uptake/clearance balance — see the Rules section below. The old bulk
% V_LN.IL2 pool was degradation-limited (~1e-7 nM), which starved both the
% division-destiny IL-2 channel and left Treg IL-2 competition inert.
IFNg = addspecies(model.Compartment(3), 'IFNg', 0, 'InitialAmountUnits', 'nanomolarity');
    set(IFNg, 'Notes', 'IFNg in the tumor compartment');
% LN-local IFNg milieu (mechanism #9, m1-A23). The tumor-draining LN has its own
% IFNg, produced by effector T cells priming there -- distinct from tumor IFNg.
% Used by the APC checkpoint synapse's PD-L1 upregulation (checkpoint_module.m),
% which previously keyed off V_T.IFNg (a cross-compartment leak: tumor inflammation
% drove LN-APC PD-L1). Sources/degradation mirror the tumor IFNg block below,
% reusing the same per-cell rates (k_IFNg_Tsec/k_IFNg_Thsec/k_IFNg_deg).
IFNg_LN = addspecies(model.Compartment(4), 'IFNg', 0, 'InitialAmountUnits', 'nanomolarity');
    set(IFNg_LN, 'Notes', 'IFNg in the lymph node compartment (local priming milieu)');
TGFb = addspecies(model.Compartment(3), 'TGFb', 0, 'InitialAmountUnits', 'nanomolarity');
    set(TGFb, 'Notes', 'TGFb in the tumor compartment');

%% Parameters — IL2
p = addparameter(model, 'k_IL2_deg', params.k_IL2_deg.Value, 'ValueUnits', params.k_IL2_deg.Units, 'Notes', 'rate of IL2 degradation ');
p.UserData.Source = params.k_IL2_deg.Notes;
    p.UserData.derivedFrom = params.k_IL2_deg.derivedFrom;
p = addparameter(model, 'k_IL2_sec', params.k_IL2_sec.Value, 'ValueUnits', params.k_IL2_sec.Units, 'Notes', 'rate of IL2 secretion per activated T cell ');
p.UserData.Source = params.k_IL2_sec.Notes;
    p.UserData.derivedFrom = params.k_IL2_sec.derivedFrom;
p = addparameter(model, 'k_IL2_cons', params.k_IL2_cons.Value, 'ValueUnits', params.k_IL2_cons.Units, 'Notes', 'rate of IL2 consumption per T cell (all clones) ');
p.UserData.Source = params.k_IL2_cons.Notes;
    p.UserData.derivedFrom = params.k_IL2_cons.derivedFrom;
p = addparameter(model, 'IL2_50', params.IL2_50.Value, 'ValueUnits', params.IL2_50.Units, 'Notes', 'T cell activation half-maximal IL2 concentration ');
p.UserData.Source = params.IL2_50.Notes;
    p.UserData.derivedFrom = params.IL2_50.derivedFrom;

%% Parameters — Activation generations
p = addparameter(model, 'N_div_base', params.N_div_base.Value, 'ValueUnits', params.N_div_base.Units, 'Notes', 'number of activated T cell generations by TCR signaling only');
p.UserData.Source = params.N_div_base.Notes;
    p.UserData.derivedFrom = params.N_div_base.derivedFrom;
p = addparameter(model, 'N_div_costim', params.N_div_costim.Value, 'ValueUnits', params.N_div_costim.Units, 'Notes', 'number of activated T cell generations by co-stimulatory signaling only ');
p.UserData.Source = params.N_div_costim.Notes;
    p.UserData.derivedFrom = params.N_div_costim.derivedFrom;
p = addparameter(model, 'N_IL2_CD8', params.N_IL2_CD8.Value, 'ValueUnits', params.N_IL2_CD8.Units, 'Notes', 'maximum number of activated T cell generations due to IL2 (CD8) ');
p.UserData.Source = params.N_IL2_CD8.Notes;
    p.UserData.derivedFrom = params.N_IL2_CD8.derivedFrom;
p = addparameter(model, 'N_IL2_CD4', params.N_IL2_CD4.Value, 'ValueUnits', params.N_IL2_CD4.Units, 'Notes', 'maximum number of activated T cell generations due to IL2 (CD4) ');
p.UserData.Source = params.N_IL2_CD4.Notes;
    p.UserData.derivedFrom = params.N_IL2_CD4.derivedFrom;
% Local IL-2 niche params (quasi-steady-state supply/demand). IL2_niche is computed
% from the priming-cluster balance: secretion (autocrine CD8 + paracrine CD4-helper)
% over local uptake (priming CD8 + competing Tregs) + clearance. This REPLACES the
% bulk V_LN.IL2 pool x f_IL2_synaptic enhancement. See the IL2_niche rule below.
%   w_IL2_help    -- paracrine CD4-helper IL-2 weight vs CD8 autocrine (incl. the
%                    higher per-cell CD4 secretion). Paracrine dominates primary CD8
%                    expansion (Cell Reports 2022 PMID 35417701).
%   kappa_IL2_Treg-- Treg IL-2 uptake weight in the niche demand (Tregs are CD25-high
%                    and out-compete; local IL-2 restriction is the dominant Treg
%                    suppression of CD8, J Immunol 2011 PMID 21849684).
%   V_niche       -- aggregate priming-niche volume (<< V_LN). Shrinking the clearance
%                    volume from V_LN to V_niche moves IL-2 out of the bulk
%                    degradation-limited regime into the competition-limited regime
%                    (this is the physical replacement for f_IL2_synaptic; Busse 2010
%                    PMID 20133667, Oyler-Yaniv 2017 PMID 28389069).
p = addparameter(model, 'w_IL2_help', params.w_IL2_help.Value, 'ValueUnits', params.w_IL2_help.Units, 'Notes', 'paracrine weight of CD4-helper IL-2 vs CD8 autocrine in the IL-2 niche supply');
    p.UserData.derivedFrom = params.w_IL2_help.derivedFrom;
p = addparameter(model, 'kappa_IL2_Treg', params.kappa_IL2_Treg.Value, 'ValueUnits', params.kappa_IL2_Treg.Units, 'Notes', 'Treg IL-2 uptake weight in the niche demand (local IL-2 restriction)');
    p.UserData.derivedFrom = params.kappa_IL2_Treg.derivedFrom;
p = addparameter(model, 'V_niche', params.V_niche.Value, 'ValueUnits', params.V_niche.Units, 'Notes', 'aggregate priming-niche volume for the local IL-2 QSS (<< V_LN)');
    p.UserData.derivedFrom = params.V_niche.derivedFrom;
% C13 excludes the urelumab/CD137 arm; division destiny contains no CD137 term.

%% Parameters — IFNg
p = addparameter(model, 'k_IFNg_Tsec', params.k_IFNg_Tsec.Value, 'ValueUnits', params.k_IFNg_Tsec.Units, 'Notes', 'rate of IFNg secretion per CD8 T cell in tumor ');
p.UserData.Source = params.k_IFNg_Tsec.Notes;
    p.UserData.derivedFrom = params.k_IFNg_Tsec.derivedFrom;
p = addparameter(model, 'k_IFNg_Thsec', params.k_IFNg_Thsec.Value, 'ValueUnits', params.k_IFNg_Thsec.Units, 'Notes', 'rate of IFNg secretion per Th cell in tumor ');
p.UserData.Source = params.k_IFNg_Thsec.Notes;
    p.UserData.derivedFrom = params.k_IFNg_Thsec.derivedFrom;
p = addparameter(model, 'k_IFNg_deg', params.k_IFNg_deg.Value, 'ValueUnits', params.k_IFNg_deg.Units, 'Notes', 'IFNg degradation rate ');
p.UserData.Source = params.k_IFNg_deg.Notes;
    p.UserData.derivedFrom = params.k_IFNg_deg.derivedFrom;
p = addparameter(model, 'IFNg_50_ind', params.IFNg_50_ind.Value, 'ValueUnits', params.IFNg_50_ind.Units, 'Notes', 'Half-Maximal IFNg level for PD-L1 induction ');
p.UserData.Source = params.IFNg_50_ind.Notes;
    p.UserData.derivedFrom = params.IFNg_50_ind.derivedFrom;

%% Parameters — TGFb
p = addparameter(model, 'k_TGFb_Tsec', params.k_TGFb_Tsec.Value, 'ValueUnits', params.k_TGFb_Tsec.Units, 'Notes', 'TGFb secretion rate per Treg cell (all clones) ');
p.UserData.Source = params.k_TGFb_Tsec.Notes;
    p.UserData.derivedFrom = params.k_TGFb_Tsec.derivedFrom;
p = addparameter(model, 'k_TGFb_deg', params.k_TGFb_deg.Value, 'ValueUnits', params.k_TGFb_deg.Units, 'Notes', 'TGFb degradation rate ');
p.UserData.Source = params.k_TGFb_deg.Notes;
    p.UserData.derivedFrom = params.k_TGFb_deg.derivedFrom;
p = addparameter(model, 'TGFb_baseline', params.TGFb_baseline.Value, 'ValueUnits', params.TGFb_baseline.Units, 'Notes', 'Baseline TGFb level in PDAC tumor ');
p.UserData.Source = params.TGFb_baseline.Notes;
    p.UserData.derivedFrom = params.TGFb_baseline.derivedFrom;
% Context-specific TGFb EC50s (mid-tier #13: split the generic TGFb_50 that
% conflated Th->Treg differentiation, M1->M2 polarization, and chemoresistance,
% matching the rest of the TGFb-50 family). All default to the Robertson-Tessi
% 2012 value (0.07 nM) so behaviour is unchanged; each is now independently
% calibratable. TGFb_50_M2pol is consumed by macrophage_module's phi_M2_skew.
p = addparameter(model, 'TGFb_50_Th2Treg', params.TGFb_50_Th2Treg.Value, 'ValueUnits', params.TGFb_50_Th2Treg.Units, 'Notes', 'Half-maximal TGFb for Th-to-Treg differentiation');
p.UserData.Source = params.TGFb_50_Th2Treg.Notes;
    p.UserData.derivedFrom = params.TGFb_50_Th2Treg.derivedFrom;
p = addparameter(model, 'TGFb_50_M2pol', params.TGFb_50_M2pol.Value, 'ValueUnits', params.TGFb_50_M2pol.Units, 'Notes', 'Half-maximal TGFb for M1-to-M2 macrophage polarization');
p.UserData.Source = params.TGFb_50_M2pol.Notes;
    p.UserData.derivedFrom = params.TGFb_50_M2pol.derivedFrom;
p = addparameter(model, 'TGFb_50_chemores', params.TGFb_50_chemores.Value, 'ValueUnits', params.TGFb_50_chemores.Units, 'Notes', 'Half-maximal TGFb for chemoresistance development (nab-paclitaxel)');
p.UserData.Source = params.TGFb_50_chemores.Notes;
    p.UserData.derivedFrom = params.TGFb_50_chemores.derivedFrom;
p = addparameter(model, 'TGFb_50_Teff', params.TGFb_50_Teff.Value, 'ValueUnits', params.TGFb_50_Teff.Units, 'Notes', 'Half-Maximal TGFb level for CD8 T cell inhibition ');
p.UserData.Source = params.TGFb_50_Teff.Notes;
    p.UserData.derivedFrom = params.TGFb_50_Teff.derivedFrom;
p = addparameter(model, 'TGFb_50_Th_exh', params.TGFb_50_Th_exh.Value, 'ValueUnits', params.TGFb_50_Th_exh.Units, 'Notes', 'Half-Maximal TGFb level for CD4 helper T cell exhaustion');
p.UserData.Source = params.TGFb_50_Th_exh.Notes;
    p.UserData.derivedFrom = params.TGFb_50_Th_exh.derivedFrom;
% Bioactive fraction scalar: scales every total-TGFb secretion rate into the
% bioactive pool represented by V_T.TGFb. Lets the EC50 family (calibrated
% against bioactive recombinant TGFb) act consistently on the V_T.TGFb
% species without renaming or splitting.
p = addparameter(model, 'f_active_TGFb', params.f_active_TGFb.Value, 'ValueUnits', params.f_active_TGFb.Units, 'Notes', 'Bioactive fraction of secreted TGFb');
p.UserData.Source = params.f_active_TGFb.Notes;
    p.UserData.derivedFrom = params.f_active_TGFb.derivedFrom;

%% Rules — Local IL-2 niche (quasi-steady-state)
% IL-2 is fast (free t1/2 ~minutes; receptor uptake by CD25-high cells faster
% still) relative to the ~day division-programming window and the cell-population
% timescale it feeds, so it is slaved to the (slow) cell counts at QSS — carried
% as an algebraic niche concentration rather than a stiff dynamic species.
%
% Priming-cluster balance: secretion = local uptake + clearance, with uptake
% LINEARIZED via the receptor Km (valid while IL2_niche <~ IL2_50; upgrade to full
% Michaelis-Menten if the niche runs hot). At low occupancy the per-cell uptake is
% (Vmax/Km)*C = (k_IL2_cons/IL2_50)*IL2_niche, so:
%     k_IL2_sec*(w_IL2_help*aTh + aCD8)
%        =  (k_IL2_cons/IL2_50)*(aCD8 + kappa_IL2_Treg*Treg)*IL2_niche
%           + k_IL2_deg*V_niche*IL2_niche
% =>  IL2_niche = k_IL2_sec*supply / ( (k_IL2_cons/IL2_50)*demand + k_IL2_deg*V_niche ).
% Both denominator terms are now volume/time (uptake: nmol/cell/hr / nM * cell =
% L/hr; clearance: 1/min * mL), so IL2_niche resolves to a concentration. Supply =
% autocrine CD8 + paracrine CD4-helper IL-2 (paracrine dominates primary CD8
% expansion, Cell Reports 2022 PMID 35417701); demand = priming CD8 + Tregs
% competing for it (local IL-2 restriction is the dominant Treg suppression of CD8,
% J Immunol 2011 PMID 21849684). The V_niche clearance volume (<< V_LN) is the
% physical replacement for the old f_IL2_synaptic*V_LN.IL2 bulk-niche estimate. When
% V_niche is small enough that uptake dominates clearance, V_niche drops out and
% IL2_niche -> (k_IL2_sec/k_IL2_cons)*IL2_50*(supply/demand) -- the anchored
% competition-ratio limit (Treg competition fully active).
p = addparameter(model, 'IL2_supply', 1, 'ValueUnits', 'cell', 'ConstantValue', false, 'Notes', 'IL-2 niche supply: autocrine CD8 + paracrine CD4-helper (weighted cell count)');
addrule(model, 'IL2_supply = w_IL2_help*V_LN.aTh + V_LN.aCD8', 'repeatedAssignment');
p = addparameter(model, 'IL2_demand', 1, 'ValueUnits', 'cell', 'ConstantValue', false, 'Notes', 'IL-2 niche demand: priming CD8 + competing Tregs (weighted cell count)');
addrule(model, 'IL2_demand = V_LN.aCD8 + kappa_IL2_Treg*V_LN.Treg', 'repeatedAssignment');
p = addparameter(model, 'IL2_niche', 1, 'ValueUnits', 'nanomolarity', 'ConstantValue', false, 'Notes', 'Local priming-niche IL-2 concentration at QSS (drives division destiny N_aT/N_aTh)');
addrule(model, 'IL2_niche = k_IL2_sec*IL2_supply/(k_IL2_cons*IL2_demand/IL2_50 + k_IL2_deg*V_niche)', 'repeatedAssignment');

%% Rules — Activation generations
% Division destiny integrates additive costim signals (Marchingo 2014): TCR
% baseline + CD28 + IL-2. The IL-2 contribution is a Hill of the local niche
% concentration IL2_niche against the sensed-concentration EC50 IL2_50 (de Jong
% 1998). C13 excludes the 4-1BB/urelumab arm.
%
% CD8 costim term is licensed by CD4 help (*H_div_help): productive CD28-costim
% that programs division destiny requires CD40L licensing of the cross-presenting
% cDC1 by helped CD4 (Bevan/Janssen/Schoenberger licensed-DC). The bare H_CD28_APC
% B7-CD28 Hill is saturated (~0.98) and treatment-invariant at the cold-tumor
% baseline, so on its own the costim term fires as hard at diagnosis as under
% vaccine (no fold). The licensing gate is H_div_help (Th_module), a DEDICATED Hill
% on V_LN.Th against a higher EC50 (Th_50_div) -- NOT the priming-throughput
% H_CD4_help, which does double duty gating cDC1->CD8 priming and sits saturated
% (~0.999), so it cannot serve as a vaccine-cold costim gate (a trace confirmed it
% is flat, 0.999->1.00). H_div_help is genuinely vaccine-cold (~0.61 at diagnosis,
% rising to ~0.92 under GVAX) so licensing makes the costim divisions a treatment
% differential (cd8_fc deficit fix, Path B; see
% notes/calibration/cd8_fc_vaccine_expansion_deficit.md).
%
% CRITICAL DEPENDENCY: H_div_help only swings the RIGHT way because the CD4 help
% arm now responds to vaccine -- which required CD4-arm epitope spreading
% (gvax_module Path A, f_epitope_spread). WITHOUT it the naive-CD4 pool craters
% and V_LN.Th *drops* under vaccine (0.76x), so H_div_help would move backwards
% (0.61->0.54) and N_aT would FALL. With CD4 spreading, V_LN.Th rises ~7x and the
% gate fires forward. N_aTh (CD4) keeps the unlicensed bare-H_CD28_APC form --
% helper division destiny is not CD40L-gated.
%
% CAVEAT (cleaner long-term form): the *truly* faithful fix is to make
% H_CD28_APC itself licensing-dependent — B7 (CD80/CD86) is upregulated by CD40L,
% so the synapse Hill should not sit saturated (~0.98) in a help-poor cold tumor.
% Multiplying the costim term by H_CD4_help here is a shortcut to the same
% effective-costim-is-licensing-dependent behavior; the deeper form scales the B7
% terms inside H_CD28_APC by H_CD4_help (see the matching note in checkpoint_module).
p = addparameter(model, 'N_aT', 1, 'ValueUnits', 'dimensionless', 'ConstantValue', false, 'Notes', 'Number of Activated CD8+ T Cell Generations (see Rules)');
% C08 reduction (2026-08-24): the separate APC-side nivolumab-sensitive
% multiplier (1-alpha_PD1_costim*H_PD1_APC) was retired because no observable
% uniquely identifies priming-side checkpoint relief apart from the retained
% tumor PD-1 and Tpex mechanisms. Its nominal untreated suppression was folded
% into N_div_costim (3 -> 1.5105), preserving basal division destiny while
% removing the extra drug-response claim. Updated-sampler refit must recover
% the GVAX+nivolumab composition contrasts; if it cannot, the fixed Hui-2017
% PD-1->CD28 coupling is the smallest layer to restore.
addrule(model, 'N_aT = N_div_base + N_div_costim*H_CD28_APC*H_div_help + N_IL2_CD8*IL2_niche/(IL2_50+IL2_niche)', 'repeatedAssignment');

p = addparameter(model, 'N_aTh', 1, 'ValueUnits', 'dimensionless', 'ConstantValue', false, 'Notes', 'Number of Activated T Helper Cell Generations (see Rules)');
addrule(model, 'N_aTh = N_div_base + N_div_costim*H_CD28_APC + N_IL2_CD4*IL2_niche/(IL2_50+IL2_niche)', 'repeatedAssignment');

%% Rules — TGFb Hill functions
% Context-specific TGFb Hills (mid-tier #13). M1->M2 polarization inlines its
% Hill in macrophage_module's phi_M2_skew (TGFb_50_M2pol), so only the Th->Treg
% and chemoresistance Hills are named here.
p = addparameter(model, 'H_TGFb_Th2Treg', 1, 'ValueUnits', 'dimensionless', 'ConstantValue', false, 'Notes', 'Hill function of TGFb for Th-to-Treg differentiation');
addrule(model, 'H_TGFb_Th2Treg = V_T.TGFb/(V_T.TGFb+TGFb_50_Th2Treg)', 'repeatedAssignment');
p = addparameter(model, 'H_TGFb_chemores', 1, 'ValueUnits', 'dimensionless', 'ConstantValue', false, 'Notes', 'Hill function of TGFb for chemoresistance development');
addrule(model, 'H_TGFb_chemores = V_T.TGFb/(V_T.TGFb+TGFb_50_chemores)', 'repeatedAssignment');

p = addparameter(model, 'H_TGFb_Teff', 1, 'ValueUnits', 'dimensionless', 'ConstantValue', false, 'Notes', 'Hill function of TGFb for Teff inhibition and chemoresistance development');
addrule(model, 'H_TGFb_Teff = V_T.TGFb/(V_T.TGFb+TGFb_50_Teff)', 'repeatedAssignment');

p = addparameter(model, 'H_TGFb_Th_exh', 1, 'ValueUnits', 'dimensionless', 'ConstantValue', false, 'Notes', 'Hill function of TGFb for CD4 helper T cell exhaustion (PD-1+, dysfunction)');
addrule(model, 'H_TGFb_Th_exh = V_T.TGFb/(V_T.TGFb+TGFb_50_Th_exh)', 'repeatedAssignment');

%% ========================================================================
%  CLONE-SPECIFIC REACTIONS
%  Processes performed only by the antigen-specific clone tracked in the
%  model. These cells were activated by specific antigen presentation and
%  secrete cytokines as part of their effector program. Rate expressions use
%  clone-level species counts directly — no diversity scaling.
%  ========================================================================

% IL-2 secretion is no longer a reaction into a bulk pool — it is folded into the
% IL2_niche QSS (k_IL2_sec*IL2_supply in the niche rule above).
% IFNg secretion by CD8 T cells in tumor (tumor-reactive effectors)
reaction = addreaction(model, 'null -> V_T.IFNg');
    set(reaction, 'ReactionRate', 'k_IFNg_Tsec*V_T.CD8');
    set(reaction, 'Notes', 'IFNg secretion from CD8 T cells in tumor (clone-specific effectors)');
% IFNg secretion by Th cells in tumor (tumor-reactive Th1 cells)
reaction = addreaction(model, 'null -> V_T.IFNg');
    set(reaction, 'ReactionRate', 'k_IFNg_Thsec*V_T.Th');
    set(reaction, 'Notes', 'IFNg secretion from Th cells in tumor (clone-specific effectors)');
% IFNg secretion by effector T cells in the LN (mechanism #9): LN-local milieu for
% APC PD-L1 upregulation during priming. Mirrors the tumor sources, same per-cell rates.
reaction = addreaction(model, 'null -> V_LN.IFNg');
    set(reaction, 'ReactionRate', 'k_IFNg_Tsec*V_LN.CD8');
    set(reaction, 'Notes', 'IFNg secretion from CD8 effector T cells in the lymph node');
reaction = addreaction(model, 'null -> V_LN.IFNg');
    set(reaction, 'ReactionRate', 'k_IFNg_Thsec*V_LN.Th');
    set(reaction, 'Notes', 'IFNg secretion from Th effector T cells in the lymph node');

%% ========================================================================
%  SHARED REACTIONS
%  Cytokine production, consumption, and degradation reactions centralized
%  from the individual T-cell modules. Rate expressions match the original
%  module implementations exactly (no diversity scaling).
%
%  NOTE: Diversity scaling (multiplying by div_CD8/div_Treg to convert
%  clone-level counts to total cell counts) is a candidate structural
%  change for a future commit. It would require simultaneous parameter
%  rescaling to maintain behavioral equivalence.
%  ========================================================================

% IL-2 degradation and consumption are folded into the IL2_niche QSS denominator
% (k_IL2_deg*V_niche clearance + k_IL2_cons*IL2_demand uptake in the niche rule above).
% IFNg degradation (concentration-dependent, no cell counts)
reaction = addreaction(model, 'V_T.IFNg -> null');
    set(reaction, 'ReactionRate', 'k_IFNg_deg*V_T.IFNg');
    set(reaction, 'Notes', 'IFNg degradation in tumor');
reaction = addreaction(model, 'V_LN.IFNg -> null');
    set(reaction, 'ReactionRate', 'k_IFNg_deg*V_LN.IFNg');
    set(reaction, 'Notes', 'IFNg degradation in lymph node (mechanism #9)');

% V_T.TGFb represents the bioactive-equivalent TGFb pool. Every
% secretion-rate parameter anchored to total-TGFb measurements (Ivanović
% 2003, Liyanage 2002 via Yang 2013, Matsuo 2000, Bachem 1992) is scaled
% by f_active_TGFb so the EC50 family (calibrated against bioactive
% recombinant TGFb) acts on a physically consistent species.

% TGFb baseline bioactive production by cancer cells + explicit first-order
% degradation. These two reactions together drive V_T.TGFb toward its set-point
% f_active_TGFb*TGFb_baseline + (other secretion)/(k_TGFb_deg*V_T). They were
% previously written as a single negative-capable relaxation source
% 'null -> V_T.TGFb' at k_TGFb_deg*(f_active_TGFb*TGFb_baseline - V_T.TGFb)*V_T,
% which hid the degradation term inside the source and could go negative (audit
% V8 flagged it as "no explicit degradation" -- a false positive: the sink was
% embedded). Split here into a constant source + an explicit V_T.TGFb -> null
% sink. This is mathematically IDENTICAL (the source minus the first-order sink
% expands to the same rate) but makes the degradation explicit and removes the
% negative-capable null-> pattern. The sink acts on the FULL TGFb pool (baseline
% + Treg + myCAF secretion), so TGFb remains bounded. TGFb_baseline is the
% literature total; scaling by f_active converts to bioactive-equivalent baseline.
reaction = addreaction(model, 'null -> V_T.TGFb');
    set(reaction, 'ReactionRate', 'k_TGFb_deg*f_active_TGFb*TGFb_baseline*V_T');
    set(reaction, 'Notes', 'Bioactive TGFb baseline production by PDAC cancer cells (scaled by f_active_TGFb).');
reaction = addreaction(model, 'V_T.TGFb -> null');
    set(reaction, 'ReactionRate', 'k_TGFb_deg*V_T.TGFb*V_T');
    set(reaction, 'Notes', 'First-order TGFb degradation (acts on the full bioactive TGFb pool: baseline + Treg + myCAF secretion).');
% TGFb secretion by Tregs (bioactive fraction of total Liyanage 2002 rate)
reaction = addreaction(model, 'null -> V_T.TGFb');
    set(reaction, 'ReactionRate', 'f_active_TGFb*k_TGFb_Tsec*V_T.Treg');
    set(reaction, 'Notes', 'Bioactive TGFb secretion by Tregs in tumor (scaled by f_active_TGFb).');

end
