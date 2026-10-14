% Cancer Module
%
% Models cancer cell growth under baseline conditions assuming logistic
% growth and first-order death
%
% Inputs: model        -- simbio model object with tumour compartment, T
%         params       -- object containing model parameter Values, Units, and Notes:
%
% Outputs: model -- SimBiology model object with new cancer module



function model = cabozantinib_module(model,p)

% Cabo Pharmacokinetics
model = cabozantinib_pk_module(model,p);

% Tumor Compartment
V_T = model.Compartment(3);

% Add Species
a = sbioselect(model,'Type','Species','Name','a');

% Add Parameters
% EC50 = addparameter(model,'EC50'    ,p.EC50_cabo.Value     ,'ValueUnits',p.EC50_cabo.Units);
IC50_MET = addparameter(model,'IC50_MET',p.IC50_MET.Value     ,'ValueUnits',p.IC50_MET.Units, 'Notes', 'IC50 for Cabozantinib_MET Complex');
IC50_RET = addparameter(model,'IC50_RET',p.IC50_RET.Value     ,'ValueUnits',p.IC50_RET.Units, 'Notes', 'IC50 for Cabozantinib_RET Complex');
IC50_AXL = addparameter(model,'IC50_AXL',p.IC50_AXL.Value     ,'ValueUnits',p.IC50_AXL.Units, 'Notes', 'IC50 for Cabozantinib_AXL Complex');
IC50_VEGFR2 = addparameter(model,'IC50_VEGFR2',p.IC50_VEGFR2.Value,'ValueUnits',p.IC50_VEGFR2.Units, 'Notes', 'IC50 for Cabozantinib_VEGFR2 Complex');

%lamC = addparameter(model,'lambda_C',p.lambda_C_cabo.Value ,'ValueUnits',p.lambda_C_cabo.Units);
%       set(lamC,'Notes','Effect Level for Cancer Cell Growth Inhibition');
%lamq = addparameter(model,'lambda_q',p.lambda_q_cabo.Value ,'ValueUnits',p.lambda_q_cabo.Units);
%       set(lamq,'Notes','Effect Level for Enhanced Vascular Transport');

k_C_resist = addparameter(model,'k_C_resist',p.k_C_resist.Value,'ValueUnits',p.k_C_resist.Units,'ConstantValue',false, 'Notes', 'transition rate to resistance clone (resemble to cancer stem cell' );

k_K_cabo = addparameter(model,'k_K_cabo',p.k_K_cabo.Value,'ValueUnits',p.k_K_cabo.Units,'ConstantValue',false, 'Notes', 'Inhibition rate of maximal tumor capacity by cabozantinib');
% 1) Resistance to Cabozantinib
R = addparameter(model,'R',1 ,'ValueUnits','dimensionless','ConstantValue',false);
addrule(model,'R = V_T.a/(V_T.a+IC50_AXL)','repeatedAssignment');

reaction = addreaction(model,'V_T.C1 -> V_T.C2');
    set(reaction,'ReactionRate','k_C_resist*V_T.C1*R');
    set(reaction,'Notes','Cancer cell resistance to cabozantinib');

reaction = addreaction(model,'V_T.C1 -> V_T.C2');
    set(reaction,'ReactionRate','k_C_resist*V_T.C1');
    set(reaction,'Notes','Cancer cell resistance to cabozantinib');

% 2) Angiogenesis
% Update vasculature by Nab-Paclitaxel
p = addparameter(model,'H_therapy_cabo',0,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'cabozantinib theraputic effect of vasculature');
addrule(model,'H_therapy_cabo = V_T.a/(V_T.a+IC50_VEGFR2)','repeatedAssignment');
for i = 1:length(model.reaction)
    % Update Th-to-Treg transdifferentiation
    if strcmp(model.reaction(i).reaction, 'null -> V_T.K')
        model.reaction(i).ReactionRate = [model.reaction(i).ReactionRate, '*(1-k_K_cabo*H_therapy_cabo)'];
    end
end

% note: had to multiply/divide by cell (value:1,units:cell) since the base
% of non-integer powers need to be dimensionless

% 3) Cancer Growth % -- Variable Cancer Cell Capacity (indirect effect from vascular normalization)
logisticTerm = sbioselect(model,'Type','Reaction','Where','ReactionRate','regexp','k_C1_growth');
for i = 1:length(logisticTerm)
     logisticTerm(i).ReactionRate = [logisticTerm(i).ReactionRate '*(1-k_K_cabo*(V_T.a)/(V_T.a+IC50_MET+IC50_RET))'];
end
%addparameter(model, 'frac',0,'ValueUnits','dimensionless','ConstantValue',false);
%addrule(model, 'frac = (V_T.a)/(V_T.a+IC50_MET)','repeatedAssignment');


% 4) Transport (indirect effect from vascular normalization)
%transportTerm = sbioselect(model,'Type','Reaction','Where','Reaction',...
%    'regexp','V_C.\w+ <?-> V_T.\w+');
%fac = '(1+lambda_q*V_T.a/(V_T.a+IC50_VEGFR2))';
%for i = 1:length(transportTerm)
%    transportTerm(i).ReactionRate = [transportTerm(i).ReactionRate '*' fac];
%end

% Rename Variables
for i = 1:length(a)
    rename(a(i),'cabozantinib');
end

%rename(a0  ,'cabo0');
%rename(lamC,'lambda_C_cabo');
%rename(lamq,'lambda_q_cabo');
rename(R   ,'R_cabo');
