% MDSC Module
%
% Inputs: model        -- simbio model object with four compartments
%         species      - MDSC--Myeloid-derived suppressor cell
%                      - ArgI--Arginase I
%                      - CCL2--Monocyte Chemoattractant Protein-1 (CCL2)
%                      - NO--Nitrite Oxide
%         params
%                      - k_MDSC_rec--MDSC recruitment rate by MCP-1
%                      - k_MDSC_death--MDSC death rate
%                      - k_CCL2_deg--Degradation rate of MCP-1
%                      - k_NO_deg--Degradation rate of NO
%                      - k_ArgI_deg--Degradation rate of Arg I
%                      - k_CCL2_sec--Secretion rate of MCP-1
%                      - k_NO_sec--Secretion rate of NO
%                      - k_ArgI_sec--Secretion rate of Arg I
%                      - IC50_inostat_C--Effective concentration of -inostat on inhibition of cancer proliferation
%                      - IC50_inostat_NO--Effective concentration of -inostat on inhibition of NO production
%                      - IC50_inostat_ArgI--Effective concentration of -inostat on inhibition of Arg I production
%                      - IC50_inostat_CCL2--Effective concentration of -inostat on inhibition of MCP-1 production
%                      - ArgI_50_Teff--Effective concentration of Arg I on inhibition of Teff activity
%                      - NO_50_Teff--Effective concentration of NO on inhibition of Teff activity
%                      - CCL2_50--Effective concentration of MCP-1 on recruitment of MDSC
%                      - ArgI_50_Treg--Effective concentration of Arg I on Th-to-Treg transdifferentiation
%
% Outputs: model -- SimBiology model object with new MDSC module


function model = MDSC_module(model,params,varargin)

% Optional Inputs
in = inputParser;
addOptional(in,'cancer_types',{'C1'});
addParameter(in,'inostat',1);
addParameter(in,'drugName','entinostat');
% Parse Inputs
parse(in,varargin{:});
cancer_types = in.Results.cancer_types;
inostat = in.Results.inostat;
drugName = in.Results.drugName;

% Species Names
species_name = 'MDSC';

% Add Species
% MDSC in tumor
MDSC = addspecies(model.Compartment(3),'MDSC',0,'InitialAmountUnits','cell', 'Notes', ['Number of MDSCs in the tumour compartment']);
% add NO
NO = addspecies(model.Compartment(3),'NO',0,'InitialAmountUnits','nanomolarity', 'Notes', 'Concentration of NO in the tumor compartment');
% add ArgI
ArgI = addspecies(model.Compartment(3),'ArgI',0,'InitialAmountUnits','nanomolarity', 'Notes', 'Concentration of Arg I in the tumor compartment');

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
    % CCL2 secretion by MDSCs and Cancer cells
    reaction = addreaction(model,'null -> V_T.CCL2');
        set(reaction,'ReactionRate','k_CCL2_sec*C_total');
        set(reaction,'Notes','CCL2 secretion by MDSCs and cancer cells');
    % CCL2 Degradation
    reaction = addreaction(model,'V_T.CCL2 -> null');
        set(reaction,'ReactionRate','k_CCL2_deg*V_T.CCL2');
        set(reaction,'Notes','CCL2 degradation');
catch
    first_call = false;
end

% Add Parameters
k_MDSC_rec = addparameter(model,'k_MDSC_rec',params.k_MDSC_rec.Value,'ValueUnits',params.k_MDSC_rec.Units, 'Notes', 'Rate of MDSC recruitment into the tumor ');
k_MDSC_rec.UserData.Source = params.k_MDSC_rec.Notes;
    k_MDSC_rec.UserData.derivedFrom = params.k_MDSC_rec.derivedFrom;
k_MDSC_death = addparameter(model,'k_MDSC_death',params.k_MDSC_death.Value,'ValueUnits',params.k_MDSC_death.Units, 'Notes', 'Rate of MDSC death ');
k_MDSC_death.UserData.Source = params.k_MDSC_death.Notes;
    k_MDSC_death.UserData.derivedFrom = params.k_MDSC_death.derivedFrom;

p = addparameter(model,'k_NO_deg',params.k_NO_deg.Value,'ValueUnits',params.k_NO_deg.Units, 'Notes', 'rate of NO degradation ');
p.UserData.Source = params.k_NO_deg.Notes;
    p.UserData.derivedFrom = params.k_NO_deg.derivedFrom;
p = addparameter(model,'k_ArgI_deg',params.k_ArgI_deg.Value,'ValueUnits',params.k_ArgI_deg.Units, 'Notes', 'rate of ArgI degradation ');
p.UserData.Source = params.k_ArgI_deg.Notes;
    p.UserData.derivedFrom = params.k_ArgI_deg.derivedFrom;

p = addparameter(model,'k_NO_sec',params.k_NO_sec.Value,'ValueUnits',params.k_NO_sec.Units, 'Notes', 'rate of NO secretion from MDSCs ');
p.UserData.Source = params.k_NO_sec.Notes;
    p.UserData.derivedFrom = params.k_NO_sec.derivedFrom;
p = addparameter(model,'k_ArgI_sec',params.k_ArgI_sec.Value,'ValueUnits',params.k_ArgI_sec.Units, 'Notes', 'rate of ArgI secretion ');
p.UserData.Source = params.k_ArgI_sec.Notes;
    p.UserData.derivedFrom = params.k_ArgI_sec.derivedFrom;

p = addparameter(model,'ArgI_50_Teff',params.ArgI_50_Teff.Value,'ValueUnits',params.ArgI_50_Teff.Units, 'Notes', 'EC50 for ArgI-mediated T cell suppression');
p.UserData.Source = params.ArgI_50_Teff.Notes;
    p.UserData.derivedFrom = params.ArgI_50_Teff.derivedFrom;
p = addparameter(model,'NO_50_Teff',params.NO_50_Teff.Value,'ValueUnits',params.NO_50_Teff.Units, 'Notes', 'EC50 for NO-mediated T cell suppression');
p.UserData.Source = params.NO_50_Teff.Notes;
    p.UserData.derivedFrom = params.NO_50_Teff.derivedFrom;
p = addparameter(model,'ArgI_50_Treg',params.ArgI_50_Treg.Value,'ValueUnits',params.ArgI_50_Treg.Units, 'Notes', 'Half-maximal ArgI level of Treg expansion ');
p.UserData.Source = params.ArgI_50_Treg.Notes;
    p.UserData.derivedFrom = params.ArgI_50_Treg.derivedFrom;

% Add Reactions
% Recruitment of MDSC (Huang 2006, PMID: 17257744)
% Enhanced by IL-6 in PDAC (Mace et al. 2018)
reaction = addreaction(model,'null -> V_T.MDSC');
    set(reaction,'ReactionRate','k_MDSC_rec*V_T*(V_T.CCL2/(V_T.CCL2 + CCL2_50))*(1 + H_IL6_MDSC)');
    set(reaction,'Notes','Recruitment of MDSC to tumor (enhanced by IL-6)');
% MDSC Death
reaction = addreaction(model,'V_T.MDSC -> null');
    set(reaction,'ReactionRate','k_MDSC_death*V_T.MDSC');
    set(reaction,'Notes','MDSC death in the tumor compartment');
% MDSC death upon tumor eradication
reaction = addreaction(model,'V_T.MDSC -> null');
    set(reaction,'ReactionRate','k_cell_clear*V_T.MDSC*(K_C2_recruit/(C_total^2 + K_C2_recruit))');
    set(reaction,'Notes','MDSC death upon tumor eradication');

% NO Degradation
reaction = addreaction(model,'V_T.NO -> null');
    set(reaction,'ReactionRate','k_NO_deg*V_T.NO');
    set(reaction,'Notes','NO degradation');
% ArgI Degradation
reaction = addreaction(model,'V_T.ArgI -> null');
    set(reaction,'ReactionRate','k_ArgI_deg*V_T.ArgI');
    set(reaction,'Notes','ArgI degradation');

% NO Secretion by MDSC
reaction = addreaction(model,'null -> V_T.NO');
    set(reaction,'ReactionRate','k_NO_sec*V_T.MDSC');
    set(reaction,'Notes','NO secretion from MDSC');
% ArgI Secretion by MDSC
reaction = addreaction(model,'null -> V_T.ArgI');
    set(reaction,'ReactionRate','k_ArgI_sec*V_T.MDSC');
    set(reaction,'Notes','ArgI secretion from MDSC');

% Set Default Hill Function for MDSC and -inostat
H_NO = addparameter(model,'H_NO',1,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function for NO-mediated inhibition on Teff');
H_ArgI_Teff = addparameter(model,'H_ArgI_Teff',1,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function for ariginase I-mediated inhibition on Teff');
H_ArgI_Treg = addparameter(model,'H_ArgI_Treg',1,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function for ariginase I-mediated Th-to-Treg transdifferentiation');
H_MDSC = addparameter(model,'H_MDSC',1,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Combined MDSC suppression via NO (T cell dysfunction) and ArgI (arginine depletion)');

% IL-6-mediated enhancement of MDSC recruitment (PDAC-specific)
% Create Hill function (will be 0 unless rule is added by fibroblast module)
H_IL6_MDSC = addparameter(model,'H_IL6_MDSC',0,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function for IL-6 enhancement of MDSC recruitment');
% Rule will be added by fibroblast_module_enhanced if IL-6 exists

% Add Hill Functions
addrule(model,'H_NO = V_T.NO/(NO_50_Teff+V_T.NO)','repeatedAssignment');
addrule(model,'H_ArgI_Teff = V_T.ArgI/(ArgI_50_Teff+V_T.ArgI)','repeatedAssignment');
addrule(model,'H_ArgI_Treg = V_T.ArgI/(ArgI_50_Treg+V_T.ArgI)','repeatedAssignment');
% Combined MDSC suppression: NO causes T cell dysfunction/apoptosis, ArgI depletes arginine
% These are independent mechanisms, combined using multiplicative "OR" logic
addrule(model,'H_MDSC = 1-(1-H_NO)*(1-H_ArgI_Teff)','repeatedAssignment');

model_rules = get(model,'Rules');

% Add -inostat drug mechanisms
if (inostat)
    p = addparameter(model,'IC50_inostat_C',params.IC50_inostat_C.Value,'ValueUnits',params.IC50_inostat_C.Units, 'Notes', '-inostat concentration for half-maximal tumor cell death ');
    p.UserData.Source = params.IC50_inostat_C.Notes;
        p.UserData.derivedFrom = params.IC50_inostat_C.derivedFrom;
    p = addparameter(model,'IC50_inostat_NO',params.IC50_inostat_NO.Value,'ValueUnits',params.IC50_inostat_NO.Units, 'Notes', 'half-maximal -inostat concentration for NO inhibition ');
    p.UserData.Source = params.IC50_inostat_NO.Notes;
        p.UserData.derivedFrom = params.IC50_inostat_NO.derivedFrom;
    p = addparameter(model,'IC50_inostat_CCL2',params.IC50_inostat_CCL2.Value,'ValueUnits',params.IC50_inostat_CCL2.Units, 'Notes', 'half-maximal -inostat concentration for CCL2 inhibition ');
    p.UserData.Source = params.IC50_inostat_CCL2.Notes;
        p.UserData.derivedFrom = params.IC50_inostat_CCL2.derivedFrom;
    p = addparameter(model,'IC50_inostat_ArgI',params.IC50_inostat_ArgI.Value,'ValueUnits',params.IC50_inostat_ArgI.Units, 'Notes', 'half-maximal -inostat concentration for Arg I inhibition ');
    p.UserData.Source = params.IC50_inostat_ArgI.Notes;
        p.UserData.derivedFrom = params.IC50_inostat_ArgI.derivedFrom;

    H_inostat_C = addparameter(model,'H_inostat_C',1,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function for -inostat-mediated inhibition on cancer cell growth');
    H_inostat_ArgI = addparameter(model,'H_inostat_ArgI',1,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function for -inostat-mediated inhibition on arginase I activity ');

    addrule(model,'H_inostat_C = V_T.inostat/(V_T.inostat+IC50_inostat_C)','repeatedAssignment');
    addrule(model,'H_inostat_ArgI = V_T.inostat/(V_T.inostat+IC50_inostat_ArgI)','repeatedAssignment');

    % Add -inostat drug-mediated effects
    for i = 1:length(model.reaction)
        if strcmp(model.reaction(i).reaction, 'null -> V_T.CCL2') && isempty(strfind(model.reaction(i).reaction, 'IC50_inostat_CCL2'))
            model.reaction(i).ReactionRate = [model.reaction(i).ReactionRate, '*(1-V_T.inostat/(V_T.inostat+IC50_inostat_CCL2))'];
        elseif strcmp(model.reaction(i).reaction, 'null -> V_T.NO') && isempty(strfind(model.reaction(i).reaction, 'IC50_inostat_NO'))
            model.reaction(i).ReactionRate = [model.reaction(i).ReactionRate, '*(1-V_T.inostat/(V_T.inostat+IC50_inostat_NO))'];
        elseif strcmp(model.reaction(i).reaction, 'null -> V_T.ArgI') && isempty(strfind(model.reaction(i).reaction, 'IC50_inostat_ArgI'))
            model.reaction(i).ReactionRate = [model.reaction(i).ReactionRate, '*(1-V_T.inostat/(V_T.inostat+IC50_inostat_ArgI))'];
        end
        for j = 1:length(cancer_types)
            if strcmp(model.reaction(i).reaction, ['null -> V_T.' cancer_types{j}]) && isempty(strfind(model.reaction(i).reaction, '*(1-H_inostat_C)'))
                model.reaction(i).ReactionRate = [model.reaction(i).ReactionRate, '*(1-H_inostat_C)'];
            end
        end
    end

    % Update Hill function for MDSC-mediated inhibition
    for i = 1:length(model_rules)
        if ~isempty(strfind(model_rules(i).Rule, 'H_MDSC = ')) && isempty(strfind(model_rules(i).Rule, 'H_inostat_ArgI'))
            model_rules(i).Rule = insertAfter(model_rules(i).Rule, 'H_ArgI_Teff', '*(1-H_inostat_ArgI)');
        end
    end

    % Add Pharmacokinetics of -inostat
    params_inostat    = pk_parameters(drugName);
    model = pk_module(model,'inostat',params_inostat,'o');
end


for i = 1:length(model.reaction)
    for j = 1:length(cancer_types)
        % Update Teff-mediated killing rate
        if strcmp(model.reaction(i).reaction, ['V_T.' cancer_types{j} ' -> V_T.C_x']) ...
                && isempty(strfind(model.reaction(i).reaction, '*(1-H_MDSC)')) && ~isempty(strfind(model.reaction(i).ReactionRate, 'k_C_T'))
            model.reaction(i).ReactionRate = [model.reaction(i).ReactionRate, '*(1-H_MDSC)'];
            model_rules(5).Rule = insertAfter(model_rules(5).Rule, ['*(1-H_PD1_' cancer_types{j} ')'], '*(1-H_MDSC)');
        % Update antigen release rate
        elseif ~isempty(strfind(model.reaction(i).reaction, 'null -> V_T.P'))
            model.reaction(i).ReactionRate = insertAfter(model.reaction(i).ReactionRate, ['*(1-H_PD1_' cancer_types{j} ')'], '*(1-H_MDSC)');
        end
    end
    % Update Th-to-Treg transdifferentiation
    if strcmp(model.reaction(i).reaction, 'V_T.Th -> V_T.Treg')
        model.reaction(i).ReactionRate = [model.reaction(i).ReactionRate, '*H_ArgI_Treg'];
    end
end
