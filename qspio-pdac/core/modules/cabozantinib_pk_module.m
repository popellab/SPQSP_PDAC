% Pharmacokinetic Module of cabozantinib

% Inputs: model        -- SimBiology model object
%         p            -- structure containing model parameters
%
% Outputs: model -- SimBiology model object with new pk module

% Created: Dec 3, 2022 (Babita Verma)

function model = cabozantinib_pk_module(model,p)

% Get Compartments from Model Object
V_C = model.Compartment(1); % Central
V_P = model.Compartment(2); % Peripheral
V_T = model.Compartment(3); % Tumor
V_LN = model.compartment(4); % Lymph node

% Add Species
species_name = 'cabozantinib';
% Central
A_C = addspecies(V_C,'a',0.0,'InitialAmountUnits','nanomolarity', 'Notes', ['Concentration of ' species_name ' in central compartment']);
set(A_C,'Tag',['$[A]_{\rm ' species_name ',C}$']);
% Peripheral
A_P = addspecies(V_P,'a',0.0,'InitialAmountUnits','nanomolarity', 'Notes', ['Concentration of ' species_name ' in peripheral compartment']);
set(A_P,'Tag',['$[A]_{\rm ' species_name ',P}$']);
% Tumour
A_T = addspecies(V_T,'a',0.0,'InitialAmountUnits','nanomolarity', 'Notes', ['Concentration of ' species_name ' in tumour compartment']);
set(A_T,'Tag',['$[A]_{\rm ' species_name ',T}$']);
% Lymph Node
A_LN = addspecies(V_LN,'a',0.0,'InitialAmountUnits','nanomolarity', 'Notes', ['Concentration of ' species_name ' in lymph node compartment']);
set(A_LN,'Tag',['$[A]_{\rm ' species_name ',LN}$']);
% Absorption site
A_site1 = addspecies(V_C,'A_site1',0.0,'InitialAmountUnits','nanomolarity', 'Notes', ['Concentration of ' species_name ' from absorption at site 1']);
A_site2 = addspecies(V_C,'A_site2',0.0,'InitialAmountUnits','nanomolarity', 'Notes', ['Concentration of ' species_name ' from absorption at site 2']);

% Add Parameters
% Absorption
k_a1 = addparameter(model,'k_a1',p.k_a1_cabozantinib.Value,'ValueUnits',p.k_a1_cabozantinib.Units, 'Notes', ['Absorption rate of ' species_name ' from site 1 to central compartment' ]);
k_a2 = addparameter(model,'k_a2',p.k_a2_cabozantinib.Value,'ValueUnits',p.k_a2_cabozantinib.Units, 'Notes', ['Absorption rate of ' species_name ' from site 2 to central compartment']);
k_cln = addparameter(model,'k_cln',p.k_cln_cabozantinib.Value,'ValueUnits',p.k_cln_cabozantinib.Units, 'Notes', ['Non-linear clearance rate of ' species_name ' from central compartment']);
Kc = addparameter(model,'Kc',p.Kc_cabozantinib.Value,'ValueUnits',p.Kc_cabozantinib.Units, 'Notes', ['Half-maximal concentration of ' species_name ' in central compartment for nonlinear clearance']);
% add parameters for dose schedule
lagP1 = addparameter(model,'lagP1',p.lagP1_cabozantinib.Value,'ValueUnits',p.lagP1_cabozantinib.Units, 'Notes', ['Lag time of ' species_name ' for fast absorption into central compartment']);
lagP2 = addparameter(model,'lagP2',p.lagP2_cabozantinib.Value,'ValueUnits',p.lagP2_cabozantinib.Units, 'Notes', ['Lag time of ' species_name ' for slow absorption into central compartment']);
F = addparameter(model,'F',p.F_cabozantinib.Value,'ValueUnits',p.F_cabozantinib.Units, 'Notes', ['Bioavailability of the drug' species_name]);

% Transportation parameter
q_P = addparameter(model,'q_P',p.q_P_cabozantinib.Value,'ValueUnits',p.q_P_cabozantinib.Units, 'Notes', p.q_P_cabozantinib.Notes);
q_T = addparameter(model,'q_T',p.q_T_cabozantinib.Value,'ValueUnits',p.q_T_cabozantinib.Units, 'Notes', p.q_T_cabozantinib.Notes);
q_LN = addparameter(model,'q_LN',p.q_LN_cabozantinib.Value,'ValueUnits',p.q_LN_cabozantinib.Units, 'Notes', p.q_LN_cabozantinib.Notes);
q_LD = addparameter(model,'q_LD',p.q_LD_cabozantinib.Value,'ValueUnits',p.q_LD_cabozantinib.Units, 'Notes', p.q_LD_cabozantinib.Notes);

gamma_C = addparameter(model,'gamma_C',p.gamma_C_cabozantinib.Value,'ValueUnits',p.gamma_C_cabozantinib.Units, 'Notes', p.gamma_C_cabozantinib.Notes);
gamma_P = addparameter(model,'gamma_P',p.gamma_P_cabozantinib.Value,'ValueUnits',p.gamma_P_cabozantinib.Units, 'Notes', p.gamma_P_cabozantinib.Notes);
gamma_T = addparameter(model,'gamma_T',p.gamma_T_cabozantinib.Value,'ValueUnits',p.gamma_T_cabozantinib.Units, 'Notes', p.gamma_T_cabozantinib.Notes);
gamma_LN = addparameter(model,'gamma_LN',p.gamma_LN_cabozantinib.Value,'ValueUnits',p.gamma_LN_cabozantinib.Units, 'Notes', p.gamma_LN_cabozantinib.Notes);

% Add Reactions
reaction = addreaction(model,'V_C.A_site2 -> V_C.a');
set(reaction,'ReactionRate','F*k_a2*V_C.A_site2');
set(reaction,'Notes',['First-order Absorption of ' species_name ' from absorption site 2 to central comparment']);
reaction = addreaction(model,'V_C.A_site1 -> V_C.a');
set(reaction,'ReactionRate','F*k_a1*V_C.A_site1');
set(reaction,'Notes',['First-order absorption of ' species_name ' from absorption site 1 to central comparment']);

% Diffusive Transport: Central to Peripheral
reaction = addreaction(model,'V_C.a <-> V_P.a'); %
set(reaction,'ReactionRate','q_P*(V_C.a/gamma_C - V_P.a/gamma_P)');
set(reaction,'Notes',[species_name ' diffusive transport to peripheral compartment']);
% Diffusive Transport: Central to Tumour
reaction = addreaction(model,'V_C.a <-> V_T.a'); % q_T = perm_CT_Ab*S_CT*Tum
set(reaction,'ReactionRate','q_T*(V_C.a/gamma_C - V_T.a/gamma_T)');
set(reaction,'Notes',[species_name ' diffusive transport to tumor compartment']);
% Diffusive Transport: Central to Lymph Node
reaction = addreaction(model,'V_C.a <-> V_LN.a');
set(reaction,'ReactionRate','q_LN*(V_C.a/gamma_C - V_LN.a/gamma_LN)');
set(reaction,'Notes',[species_name ' diffusive transport to lymph node compartment']);
% Convective Transport: Tumour to Lymph Node
reaction = addreaction(model,'V_T.a -> V_LN.a');
set(reaction,'ReactionRate','q_LD*V_T.a/gamma_T');
set(reaction,'Notes',[species_name ' convective transport from tumor to lymph node']);
% Convective Transport: Lymph Node to Central
reaction = addreaction(model,'V_LN.a -> V_C.a');
set(reaction,'ReactionRate','q_LD*V_LN.a/gamma_LN');
set(reaction,'Notes',[species_name ' convective transport from lymph node to central']);
% Clearence from Central
reaction = addreaction(model,'V_C.a -> null');
set(reaction,'ReactionRate','k_cln*V_C.a/(V_C.a+Kc)');
set(reaction,'Notes',['Nonlinear clearance of ' species_name ' from central compartment']);

% Rename Objects with 'species_name'
% rename(A_C,species_name);
% rename(A_P,species_name);
% rename(A_T,species_name);
% rename(A_LN,species_name);
% rename(A_site1,[species_name '_absorption_site1']);
% rename(A_site2,[species_name '_absorption_site2']);
rename(k_a1,['k_a1_' species_name]);
rename(k_a2,['k_a2_' species_name]);
rename(lagP1,['lagP1_' species_name]);
rename(lagP2,['lagP2_' species_name]);
rename(F,['F_' species_name]);
rename(q_P,['q_P_' species_name]);
rename(q_T,['q_T_' species_name]);
rename(q_LN,['q_LN_' species_name]);
rename(q_LD,['q_LD_' species_name]);
rename(k_cln,['k_cln_' species_name]);
rename(Kc,['Kc_' species_name]);
rename(gamma_C,['gamma_C_' species_name]);
rename(gamma_P,['gamma_P_' species_name]);
rename(gamma_T,['gamma_T_' species_name]);
rename(gamma_LN,['gamma_LN_' species_name]);

end
