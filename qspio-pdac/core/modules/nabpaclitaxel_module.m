% Nab-Paclitaxel Module (trimmed 2026-07-13)
%
% Faithful nab-paclitaxel population PK (3-compartment with saturable Michaelis-
% Menten distribution + clearance) plus a plain Emax cell-kill PD on V_T.C1.
% Resistance clone (C2) and anti-angiogenic (vascular) effects were REMOVED to
% keep the PASS-01 chemo network simple and consistent with chemo_pk_module
% (resistance limited for now). Re-add from git history if needed later.
%
% Inputs:  model  -- SimBiology model object (needs cancer_module: V_T.C1/C_total/C_x)
%          params -- default parameters (Vmcl,Kcl,Q2,Vmt,Kt,BSA; r_nabp,k_C_nabp,
%                    IC50_nabp,MW_nabp,Kc_nabp)
% Outputs: model  -- model with nab-paclitaxel PK-PD

function model = nabpaclitaxel_module(model, params)

% ---- PK compartments (apparent volumes of distribution) ----
comp_V1 = addcompartment(model,'V_1',15.8,'CapacityUnits','liter','ConstantCapacity',false);
    set(comp_V1,'Notes','Central compartment (V1) for nab-paclitaxel PK');
comp_V2 = addcompartment(model,'V_2',1650,'CapacityUnits','liter','ConstantCapacity',false);
    set(comp_V2,'Notes','Peripheral compartment (V2) for nab-paclitaxel PK');
comp_V3 = addcompartment(model,'V_3',75.4,'CapacityUnits','liter','ConstantCapacity',false);
    set(comp_V3,'Notes','Peripheral compartment (V3) for nab-paclitaxel PK');

% ---- PK parameters ----
addparameter(model,'Vmcl',params.Vmcl.Value,'ValueUnits',params.Vmcl.Units, 'Notes', 'Maximum elimination rate of nab-paclitaxel from V1');
addparameter(model,'Kcl',params.Kcl.Value,'ValueUnits',params.Kcl.Units, 'Notes', 'Nab-paclitaxel conc in V1 at 50% Vmcl');
addparameter(model,'Q2',params.Q2.Value,'ValueUnits',params.Q2.Units, 'Notes', 'Intercompartmental clearance V1<->V3');
addparameter(model,'Vmt',params.Vmt.Value,'ValueUnits',params.Vmt.Units, 'Notes', 'Maximum intercompartmental distribution rate V1<->V2');
addparameter(model,'Kt',params.Kt.Value,'ValueUnits',params.Kt.Units, 'Notes', 'Nab-paclitaxel conc in V1 at 50% Vmt');
addparameter(model,'BSA',params.BSA.Value,'ValueUnits',params.BSA.Units, 'Notes', 'Body surface area');

% ---- PK species (central + 2 peripheral) ----
addspecies(comp_V1,'NabP',0,'InitialAmountUnits','nanogram/milliliter', 'Notes', 'Nab-paclitaxel in V1');
addspecies(comp_V2,'NabP',0,'InitialAmountUnits','nanogram/milliliter', 'Notes', 'Nab-paclitaxel in V2');
addspecies(comp_V3,'NabP',0,'InitialAmountUnits','nanogram/milliliter', 'Notes', 'Nab-paclitaxel in V3');

% ---- PK reactions (saturable distribution/clearance) ----
r = addreaction(model,'V_1.NabP <-> V_2.NabP');
    set(r,'ReactionRate','Vmt/(V_1.NabP+Kt)*V_1.NabP - Vmt/(V_2.NabP+Kt)*V_2.NabP');
    set(r,'Notes','Intercompartmental distribution of nab-paclitaxel (V1<->V2, saturable)');
r = addreaction(model,'V_1.NabP <-> V_3.NabP');
    set(r,'ReactionRate','Q2*V_1.NabP - Q2*V_3.NabP');
    set(r,'Notes','Intercompartmental clearance of nab-paclitaxel (V1<->V3, linear)');
r = addreaction(model,'V_1.NabP -> null');
    set(r,'ReactionRate','Vmcl/(V_1.NabP+Kcl)*V_1.NabP');
    set(r,'Notes','Clearance of nab-paclitaxel from V1 (saturable)');

% ---- PD: tumor penetration + Emax kill of V_T.C1 ----
r_nabp = addparameter(model,'r_nabp',params.r_nabp.Value,'ValueUnits',params.r_nabp.Units,'ConstantValue',false, 'Notes', 'Tumour:plasma concentration ratio of nab-paclitaxel');
    r_nabp.UserData.Source = params.r_nabp.Notes; r_nabp.UserData.derivedFrom = params.r_nabp.derivedFrom;
k_C_nabp = addparameter(model,'k_C_nabp',params.k_C_nabp.Value,'ValueUnits',params.k_C_nabp.Units,'ConstantValue',false, 'Notes', 'Cancer cell killing rate by nab-paclitaxel');
    k_C_nabp.UserData.Source = params.k_C_nabp.Notes; k_C_nabp.UserData.derivedFrom = params.k_C_nabp.derivedFrom;
IC50_nabp = addparameter(model,'IC50_nabp',params.IC50_nabp.Value,'ValueUnits',params.IC50_nabp.Units,'ConstantValue',false, 'Notes', 'Half-maximal nab-paclitaxel conc for cancer kill');
    IC50_nabp.UserData.Source = params.IC50_nabp.Notes; IC50_nabp.UserData.derivedFrom = params.IC50_nabp.derivedFrom;
MW_nabp = addparameter(model,'MW_nabp',params.MW_nabp.Value,'ValueUnits',params.MW_nabp.Units, 'Notes', 'Molecular weight of nab-paclitaxel');
    MW_nabp.UserData.Source = params.MW_nabp.Notes; MW_nabp.UserData.derivedFrom = params.MW_nabp.derivedFrom;
Kc_nabp = addparameter(model,'Kc_nabp',params.Kc_nabp.Value,'ValueUnits',params.Kc_nabp.Units, 'Notes', 'Half-maximal cancer cell number for cytotoxic drug diffusion');
    Kc_nabp.UserData.Source = params.Kc_nabp.Notes; Kc_nabp.UserData.derivedFrom = params.Kc_nabp.derivedFrom;

% Tumor concentration as a PARAMETER via assignment rule (NOT a V_T species --
% a rule-assigned conc species in the variable-volume V_T compartment trips the
% codegen amount<->conc volume conversion; see chemo_pk_module).
addparameter(model,'Ct_nabp',0,'ValueUnits','nanomolarity','ConstantValue',false, 'Notes', 'Nab-paclitaxel concentration in tumor (r_nabp x plasma / MW)');
addrule(model,'Ct_nabp = r_nabp*V_1.NabP/MW_nabp','repeatedAssignment');

r = addreaction(model,'V_T.C1 -> V_T.C_x');
    set(r,'ReactionRate','k_C_nabp*V_T.C1*(Ct_nabp/(Ct_nabp+IC50_nabp))*min(C_total,Kc_nabp)/C_total');
    set(r,'Notes','Cancer cell death by nab-paclitaxel (Emax)');

% ---- Resistant clone C2: same kill, IC50 raised r_resist-fold (cross-drug) ----
% r_resist is shared across all cytotoxic modules; add it once (first drug module).
if isempty(sbioselect(model,'Type','parameter','Name','r_resist'))
    r_res = addparameter(model,'r_resist',params.r_resist.Value,'ValueUnits',params.r_resist.Units,'ConstantValue',false, 'Notes', 'Fold-increase of cytotoxic IC50 in the resistant clone C2 (cross-drug)');
    r_res.UserData.Source = params.r_resist.Notes; r_res.UserData.derivedFrom = params.r_resist.derivedFrom;
end
r = addreaction(model,'V_T.C2 -> V_T.C_x');
    set(r,'ReactionRate','k_C_nabp*V_T.C2*(Ct_nabp/(Ct_nabp+IC50_nabp*r_resist))*min(C_total,Kc_nabp)/C_total');
    set(r,'Notes','Resistant-clone (C2) death by nab-paclitaxel (Emax, IC50 x r_resist)');

end
