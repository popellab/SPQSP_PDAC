% Nab-Paclitaxel Module
%
% Models nab-paclitaxel PK-PD
%
% Inputs: model        -- SimBiology model object
%         params       -- object containing the default parameters
%
% Outputs: model -- SimBiology model object with new nab-paclitaxel module



function model = nabpaclitaxel_module(model, params)

% Setup Compartments
comp_V1 = addcompartment(model,'V_1',15.8,'CapacityUnits','liter','ConstantCapacity',false);
    set(comp_V1,'Notes',['Central compartment (V1) for nab-paclitaxel PK']);
comp_V2 = addcompartment(model,'V_2',1650,'CapacityUnits','liter','ConstantCapacity',false);
    set (comp_V2,'Notes',['Peripheral compartment (V2) for nab-paclitaxel PK']);
comp_V3 = addcompartment(model,'V_3',75.4,'CapacityUnits','liter','ConstantCapacity',false);
    set (comp_V3,'Notes',['Peripheral compartment (V3) for nab-paclitaxel PK']);

% comp_M0 = addcompartment(model,'M_0',1,'CapacityUnits','liter','ConstantCapacity',false);
%     set(comp_M0,'Notes',['Maturation compartment (M0) ']);
% comp_M1 = addcompartment(model,'M_1',1,'CapacityUnits','liter','ConstantCapacity',false);
%     set(comp_M1,'Notes',['Maturation compartment (M1) ']);
% comp_M2 = addcompartment(model,'M_2',1,'CapacityUnits','liter','ConstantCapacity',false);
%     set(comp_M2,'Notes',['Maturation compartment (M2) ']);
% comp_M3 = addcompartment(model,'M_3',1,'CapacityUnits','liter','ConstantCapacity',false);
%     set(comp_M3,'Notes',['Maturation compartment (M3) ']);
% comp_M4 = addcompartment(model,'M_4',1,'CapacityUnits','liter','ConstantCapacity',false);
%     set(comp_M4,'Notes',['Maturation compartment (M4) ']);

% Model Parameters
Vmcl = addparameter(model,'Vmcl',params.Vmcl.Value,'ValueUnits',params.Vmcl.Units, 'Notes', 'Maximum elimination rate of nab-paclitaxel from the central compartment V1');
Kcl = addparameter(model,'Kcl',params.Kcl.Value,'ValueUnits',params.Kcl.Units, 'Notes', 'Nab-paclitaxel concentration in the central compartment V1 at 50% Vmcl');
Q2 = addparameter(model,'Q2',params.Q2.Value,'ValueUnits',params.Q2.Units, 'Notes', 'Intercompartmental clearance between the central compartment V1 and the second peripheral compartment V3');
Vmt = addparameter(model,'Vmt',params.Vmt.Value,'ValueUnits',params.Vmt.Units, 'Notes', 'Maximum intercompartmental distribution rate between the central compartment V1 and the first peripheral compartment V2');
Kt = addparameter(model,'Kt',params.Kt.Value,'ValueUnits',params.Kt.Units, 'Notes', 'Nab-paclitaxel concentration in the central compartment V1 at 50% Vmt');
BSA = addparameter(model,'BSA',params.BSA.Value,'ValueUnits',params.BSA.Units, 'Notes', 'Body surface area');

% PD Parameters
% p = addparameter(model,'E',0,'ValueUnits','dimensionless','ConstantValue',false);
%     set(p,'Notes',['Nab-paclitaxel effect on neutrophils ']);
% p = addparameter(model,'gamma',0.187,'ValueUnits','dimensionless');
%     set(p,'Notes',['Feedback parameter for nab-paclitaxel pd ']);
% p = addparameter(model,'MTT',117,'ValueUnits','hour');
%     set(p,'Notes',['Mean transit time of neutrophils ']);
% p = addparameter(model,'k',0.0342,'ValueUnits','1/hour','ConstantValue',false);
%     set(p,'Notes',['Transit rate of neutrophils ']);
% p = addparameter(model,'slope',0.00253,'ValueUnits','1/(nanogram/milliliter)');
%     set(p,'Notes',['Nab-paclitaxel effect slope ']);
% p = addparameter(model,'Neu_base',4.28e9,'ValueUnits','cell/liter');
%     set(p,'Notes',['Baseline circulating neutrophil level ']);

% Model Species
s = addspecies(comp_V1,'NabP',0,'InitialAmountUnits','nanogram/milliliter', 'Notes', 'Nab-paclitaxel in the central compartment V1');
s = addspecies(comp_V2,'NabP',0,'InitialAmountUnits','nanogram/milliliter', 'Notes', 'Nab-paclitaxel in the peripheral compartment V2');
s = addspecies(comp_V3,'NabP',0,'InitialAmountUnits','nanogram/milliliter', 'Notes', 'Nab-paclitaxel in the peripheral compartment V3');

% s = addspecies(comp_M0,'Neu',4.28e9,'InitialAmountUnits','cell/liter');
%     set(s,'Notes','Neutrophil in bone marrow');
% s = addspecies(comp_M1,'Neu',4.28e9,'InitialAmountUnits','cell/liter');
%     set(s,'Notes','Neutrophil in the maturation compartment M1');
% s = addspecies(comp_M2,'Neu',4.28e9,'InitialAmountUnits','cell/liter');
%     set(s,'Notes','Neutrophil in the maturation compartment M2');
% s = addspecies(comp_M3,'Neu',4.28e9,'InitialAmountUnits','cell/liter');
%     set(s,'Notes','Neutrophil in the maturation compartment M3');
% s = addspecies(comp_M4,'Neu',4.28e9,'InitialAmountUnits','cell/liter');
%     set(s,'Notes','Circulating neutrophil ');

% Model Reaction
r = addreaction(model,'V_1.NabP <-> V_2.NabP');
    set(r,'ReactionRate','Vmt/(V_1.NabP+Kt)*V_1.NabP - Vmt/(V_2.NabP+Kt)*V_2.NabP');
    set(r,'Notes','Intercompartmental distribution of nab-paclitaxel between V1 and V2 compartment');
r = addreaction(model,'V_1.NabP <-> V_3.NabP');
    set(r,'ReactionRate','Q2*V_1.NabP - Q2*V_3.NabP');
    set(r,'Notes','Intercompartmental clearance of nab-paclitaxel between V1 and V3 compartment');
r = addreaction(model,'V_1.NabP -> null');
    set(r,'ReactionRate','Vmcl/(V_1.NabP+Kcl)*V_1.NabP');
    set(r,'Notes','Clearance of nab-paclitaxel from V1 compartment');

% r = addreaction(model,'null -> M_0.Neu');
%     set(r,'ReactionRate','k*M_0.Neu*(1-E)*(Neu_base/M_4.Neu)^gamma');
% r = addreaction(model,'M_4.Neu -> null');
%     set(r,'ReactionRate','k*M_4.Neu');
% r = addreaction(model,'M_0.Neu -> M_1.Neu');
%     set(r,'ReactionRate','k*M_0.Neu');
% r = addreaction(model,'M_1.Neu -> M_2.Neu');
%     set(r,'ReactionRate','k*M_1.Neu');
% r = addreaction(model,'M_2.Neu -> M_3.Neu');
%     set(r,'ReactionRate','k*M_2.Neu');
% r = addreaction(model,'M_3.Neu -> M_4.Neu');
%     set(r,'ReactionRate','k*M_3.Neu');


% Model Rule
% p = addparameter(model,'age',0,'ValueUnits','dimensionless','ConstantValue',false);
%     set(p,'Notes',['Age factor to estimate paclitaxel effect slope ']);

% addrule(model,'k = 4/MTT','repeatedAssignment');
% addrule(model,'E = V_1.NabP*slope*(1+age)','repeatedAssignment');


% PD Cytotoxicity
r_nabp = addparameter(model,'r_nabp',params.r_nabp.Value,'ValueUnits',params.r_nabp.Units,'ConstantValue',false, 'Notes', 'Tumour to plasma concentration ratio of nab-paclitaxel ');
r_nabp.UserData.Source = params.r_nabp.Notes;
    r_nabp.UserData.derivedFrom = params.r_nabp.derivedFrom;
k_C_nabp = addparameter(model,'k_C_nabp',params.k_C_nabp.Value,'ValueUnits',params.k_C_nabp.Units,'ConstantValue',false, 'Notes', 'Cancer cell killing rate by nab-paclitaxel ');
k_C_nabp.UserData.Source = params.k_C_nabp.Notes;
    k_C_nabp.UserData.derivedFrom = params.k_C_nabp.derivedFrom;
IC50_nabp = addparameter(model,'IC50_nabp',params.IC50_nabp.Value,'ValueUnits',params.IC50_nabp.Units,'ConstantValue',false, 'Notes', 'Half-maximal nab-paclitaxel concentration for cancer cell killing ');
IC50_nabp.UserData.Source = params.IC50_nabp.Notes;
    IC50_nabp.UserData.derivedFrom = params.IC50_nabp.derivedFrom;
MW_nabp = addparameter(model,'MW_nabp',params.MW_nabp.Value,'ValueUnits',params.MW_nabp.Units, 'Notes', 'Molecular weight of nab-paclitaxel ');
MW_nabp.UserData.Source = params.MW_nabp.Notes;
    MW_nabp.UserData.derivedFrom = params.MW_nabp.derivedFrom;
Kc_nabp = addparameter(model,'Kc_nabp',params.Kc_nabp.Value,'ValueUnits',params.Kc_nabp.Units, 'Notes', 'Half-Maximal cancer cell number for cytotoxic drug diffusion ');
Kc_nabp.UserData.Source = params.Kc_nabp.Notes;
    Kc_nabp.UserData.derivedFrom = params.Kc_nabp.derivedFrom;

NabP = addspecies(model.Compartment(3),'NabP',0,'InitialAmountUnits','nanomolarity', 'Notes', 'Nab-paclitaxel concentration in tumour ');

addrule(model,'V_T.NabP = r_nabp*V_1.NabP/MW_nabp','repeatedAssignment');

reaction = addreaction(model,'V_T.C1 -> V_T.C_x');
    set(reaction,'ReactionRate','k_C_nabp*V_T.C1*(V_T.NabP/(V_T.NabP+IC50_nabp))*min(C_total,Kc_nabp)/C_total'); % *(1-C_total/(C_total+Kc_nabp))
    set(reaction,'Notes','Cancer cell death by nab-paclitaxel ');

%% Resistance
k_C_resist_nab = addparameter(model,'k_C_resist_nab',params.k_C_resist_nab.Value,'ValueUnits',params.k_C_resist_nab.Units,'ConstantValue',false, 'Notes', 'Cancer resistance to nab-paclitaxel ');
k_C_resist_nab.UserData.Source = params.k_C_resist.Notes;
    k_C_resist_nab.UserData.derivedFrom = params.k_C_resist.derivedFrom;
r_resist = addparameter(model,'r_resist',params.r_resist.Value,'ValueUnits',params.r_resist.Units, 'Notes', 'Number of folds increase of nab-paclitaxel EC50 in resistant cancer clones ');
r_resist.UserData.Source = params.r_resist.Notes;
    r_resist.UserData.derivedFrom = params.r_resist.derivedFrom;

reaction = addreaction(model,'V_T.C1 -> V_T.C2');
    set(reaction,'ReactionRate','k_C_resist_nab*V_T.C1*H_TGFb');
    set(reaction,'Notes','Cancer cell resistance to nab-paclitaxel ');
reaction = addreaction(model,'V_T.C2 -> V_T.C_x');
    set(reaction,'ReactionRate','k_C_nabp*V_T.C2*(V_T.NabP/(V_T.NabP+IC50_nabp*r_resist))*min(C_total,Kc_nabp)/C_total');
    set(reaction,'Notes','Resistant cancer cell death by nab-paclitaxel ');

% Get Model Rules for Updating
model_rules = get(model,'Rules');

% Update Cancer Killing by Nab-Paclitaxel
for j = 1:length(model_rules)
    if ~isempty(strfind(model_rules(j).Rule, 'k_C1_therapy_nabp')) && isempty(strfind(model_rules(j).Rule, 'k_C_nabp'))
        model_rules(j).Rule = [model_rules(j).Rule ' + k_C_nabp*(V_T.NabP/(V_T.NabP+IC50_nabp))*min(C_total,Kc_nabp)/C_total'];
    elseif ~isempty(strfind(model_rules(j).Rule, 'k_C2_therapy_nabp')) && isempty(strfind(model_rules(j).Rule, 'k_C_nabp'))
        model_rules(j).Rule = [model_rules(j).Rule ' + k_C_nabp*(V_T.NabP/(V_T.NabP+IC50_nabp*r_resist))*min(C_total,Kc_nabp)/C_total'];
    end
end

% set tumour growth rate of the resistant the same as the sensitive clone
addrule(model,'k_C2_growth = k_C1_growth','repeatedAssignment');
addrule(model,'k_C2_death = k_C1_death','repeatedAssignment');


%% Vascularization
p = addparameter(model,'k_vas_nabp',params.k_vas_nabp.Value,'ValueUnits',params.k_vas_nabp.Units,'ConstantValue',false, 'Notes', 'Secretion rate of angiogenic factors induced by nab-paclitaxel ');
p.UserData.Source = params.k_vas_nabp.Notes;
    p.UserData.derivedFrom = params.k_vas_nabp.derivedFrom;
p = addparameter(model,'IC50_nabp_vas',params.IC50_nabp_vas.Value,'ValueUnits',params.IC50_nabp_vas.Units,'ConstantValue',false, 'Notes', 'Half-maximal conc. of nab-paclitaxel on angiogenic factor induction ');
p.UserData.Source = params.IC50_nabp_vas.Notes;
    p.UserData.derivedFrom = params.IC50_nabp_vas.derivedFrom;
p = addparameter(model,'k_K_nabp',params.k_K_nabp.Value,'ValueUnits',params.k_K_nabp.Units,'ConstantValue',false, 'Notes', 'Inhibition rate of maximal tumor capacity by nab-paclitaxel ');
p.UserData.Source = params.k_K_nabp.Notes;

    p.UserData.derivedFrom = params.k_K_nabp.derivedFrom;

reaction = addreaction(model,'null -> V_T.VEGF');
    set(reaction,'ReactionRate','k_vas_nabp*C_total*V_T.NabP/(V_T.NabP+IC50_nabp_vas)');
    set(reaction,'Notes','Angiogenic factor release in response to nab-paclitaxel ');
reaction = addreaction(model,'V_T.K -> null');
    set(reaction,'ReactionRate','k_K_nabp*V_T.K*V_T.NabP');
    set(reaction,'Notes','Inhibition of tumor vasculature due to endothelial cell death by nab-paclitaxel ');

% Update vasculature by Nab-Paclitaxel
p = addparameter(model,'k_K_therapy_nabp',0,'ValueUnits','cell/day','ConstantValue',false, 'Notes', 'nab-paxlitaxel theraputic effect of vasculature');
addrule(model,'k_K_therapy_nabp = k_K_nabp*V_T.K*V_T.NabP','repeatedAssignment');
