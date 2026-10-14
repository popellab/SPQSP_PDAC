% Treg CTLA4 Module
%
% Models Treg depletion through antibody binding to CTLA4
% [Use after checkpoint and Treg module]
%
% Inputs: model        -- SimBiology model object with four compartments
%         params       -- object containing the default parameters
%
% Outputs: model -- SimBiology model object with new Treg-CTLA4 module


function model = Treg_ADCC_module(model,params)

% Determine if some of the prameters that could have been defined in Checkpoint module are defined
first_call = true;
try % Add kon Values
    kon = addparameter(model,'kon_CTLA4_aCTLA4',params.kon_CTLA4_aCTLA4.Value,'ValueUnits',params.kon_CTLA4_aCTLA4.Units, 'Notes', 'kon of CTLA4-aCTLA4 binding ');
    kon.UserData.Source = params.kon_CTLA4_aCTLA4.Notes;
    kon.UserData.derivedFrom = params.kon_CTLA4_aCTLA4.derivedFrom;
catch
    first_call = false;
end
if first_call
% add aCTLA-4 PK if it does not exist yet
params_aCTLA4  = pk_parameters('ipilimumab');
model = pk_module(model,'aCTLA4' ,params_aCTLA4);
% Add koff Values
koff = addparameter(model,'koff_CTLA4_aCTLA4',params.koff_CTLA4_aCTLA4.Value,'ValueUnits',params.koff_CTLA4_aCTLA4.Units, 'Notes', 'koff of CTLA4-aCTLA4 binding ');
koff.UserData.Source = params.koff_CTLA4_aCTLA4.Notes;
    koff.UserData.derivedFrom = params.koff_CTLA4_aCTLA4.derivedFrom;
% Bivalent anibody parameters
p = addparameter(model,'Chi_CTLA4_aCTLA4' ,params.Chi_CTLA4_aCTLA4.Value ,'ValueUnits',params.Chi_CTLA4_aCTLA4.Units, 'Notes', 'Antibody cross-arm binding efficiency ');
p.UserData.Source = params.Chi_CTLA4_aCTLA4.Notes;
    p.UserData.derivedFrom = params.Chi_CTLA4_aCTLA4.derivedFrom;
end

% Add area of a T cell if not defined before in antigen module
try
    parameter = addparameter(model,'A_Tcell' ,params.A_Tcell.Value ,'ValueUnits',params.A_Tcell.Units, 'Notes', 'Surface area of the T cell ');
    parameter.UserData.Source = params.A_Tcell.Notes;
    parameter.UserData.derivedFrom = params.A_Tcell.derivedFrom;
catch
end

% Total amount of CTLA4 on Treg
p = addparameter(model,'Treg_CTLA4_tot',params.Treg_CTLA4_tot.Value,'ValueUnits',params.Treg_CTLA4_tot.Units,'ConstantValue',false, 'Notes', 'Total number of CTLA4 on Treg cells ');
p.UserData.Source = params.Treg_CTLA4_tot.Notes;
    p.UserData.derivedFrom = params.Treg_CTLA4_tot.derivedFrom;
% CTLA4-related Hill parameters
p = addparameter(model,'Treg_CTLA4_50',params.Treg_CTLA4_50.Value,'ValueUnits',params.Treg_CTLA4_50.Units, 'Notes', 'CTLA4 occupancy for half-maximal Treg inactivation by macrophages ');
p.UserData.Source = params.Treg_CTLA4_50.Notes;
    p.UserData.derivedFrom = params.Treg_CTLA4_50.derivedFrom;
p = addparameter(model,'n_Treg_CTLA4',params.n_Treg_CTLA4.Value,'ValueUnits',params.n_Treg_CTLA4.Units, 'Notes', 'CTLA4 occupancy Hill coefficient for Treg inactivation by macrophages ');
p.UserData.Source = params.n_Treg_CTLA4.Notes;
    p.UserData.derivedFrom = params.n_Treg_CTLA4.derivedFrom;
% Treg ADCC rates
p = addparameter(model,'k_CTLA4_ADCC',params.k_CTLA4_ADCC.Value,'ValueUnits',params.k_CTLA4_ADCC.Units,'ConstantValue',false, 'Notes', 'Treg ADCC rate by CTLA4 antibody');
p.UserData.Source = params.k_CTLA4_ADCC.Notes;
    p.UserData.derivedFrom = params.k_CTLA4_ADCC.derivedFrom;
% Define Hill function for ADCC in tumor and peripheral compartments
p = addparameter(model,'H_Treg_T',0.0,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function of Treg ADCC in tumor');
p = addparameter(model,'H_Treg_P',0.0,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function of Treg ADCC in peripheral compartment');

% Species for states of CTLA4 on Treg
x = addspecies(model.Compartment(3),'Treg_CTLA4',0,'InitialAmountUnits','molecule', 'Notes', 'Number of free CTLA4 molecules on Treg in Tumour compartment');
x = addspecies(model.Compartment(3),'Treg_CTLA4_aCTLA4',0,'InitialAmountUnits','molecule', 'Notes', 'Number of CTLA4-aCTLA4 complex on Treg in Tumour compartment');
x = addspecies(model.Compartment(3),'Treg_CTLA4_aCTLA4_CTLA4',0,'InitialAmountUnits','molecule', 'Notes', 'Number of CTLA4-aCTLA4-CTLA4 complex on Treg in Tumour compartment');
x = addspecies(model.Compartment(2),'Treg_CTLA4',0,'InitialAmountUnits','molecule', 'Notes', 'Number of free CTLA4 molecules on Treg in Peripheral compartment');
x = addspecies(model.Compartment(2),'Treg_CTLA4_aCTLA4',0,'InitialAmountUnits','molecule', 'Notes', 'Number of CTLA4-aCTLA4 complex on Treg in Peripheral compartment');
x = addspecies(model.Compartment(2),'Treg_CTLA4_aCTLA4_CTLA4',0,'InitialAmountUnits','molecule', 'Notes', 'Number of CTLA4-aCTLA4-CTLA4 complex on Treg in Peripheral compartment');

% Initialize the total CTLA4 on the cells
addrule(model,'V_T.Treg_CTLA4 = Treg_CTLA4_tot' ,'initialAssignment');
addrule(model,'V_P.Treg_CTLA4 = Treg_CTLA4_tot' ,'initialAssignment');
% Add the Hill parameters and the rules for them
addrule(model,'H_Treg_T = ((V_T.Treg_CTLA4_aCTLA4+2*V_T.Treg_CTLA4_aCTLA4_CTLA4)/Treg_CTLA4_50)^n_Treg_CTLA4/(((V_T.Treg_CTLA4_aCTLA4+2*V_T.Treg_CTLA4_aCTLA4_CTLA4)/Treg_CTLA4_50)^n_Treg_CTLA4 + 1)','repeatedAssignment');
addrule(model,'H_Treg_P = ((V_P.Treg_CTLA4_aCTLA4+2*V_P.Treg_CTLA4_aCTLA4_CTLA4)/Treg_CTLA4_50)^n_Treg_CTLA4/(((V_P.Treg_CTLA4_aCTLA4+2*V_P.Treg_CTLA4_aCTLA4_CTLA4)/Treg_CTLA4_50)^n_Treg_CTLA4 + 1)','repeatedAssignment');

% Binding and unbinding of aCTLA4 to CTLA4 on Treg on Tumour and Peripheral
R = addreaction(model, 'V_T.Treg_CTLA4 <-> V_T.Treg_CTLA4_aCTLA4');
    set (R, 'ReactionRate', 'kon_CTLA4_aCTLA4*(V_T.Treg_CTLA4 * V_T.aCTLA4/gamma_T_aCTLA4) -  koff_CTLA4_aCTLA4*V_T.Treg_CTLA4_aCTLA4');
    set (R, 'Notes'       , 'binding and unbinding of CTLA4 to aCTLA4 on Treg surface in Tumor');
R = addreaction(model, 'V_T.Treg_CTLA4_aCTLA4 + V_T.Treg_CTLA4 <-> V_T.Treg_CTLA4_aCTLA4_CTLA4');
    set (R, 'ReactionRate', 'Chi_CTLA4_aCTLA4*kon_CTLA4_aCTLA4*(V_T.Treg_CTLA4 * V_T.Treg_CTLA4_aCTLA4)/A_Tcell -  koff_CTLA4_aCTLA4*V_T.Treg_CTLA4_aCTLA4_CTLA4');
    set (R, 'Notes'       , 'binding and unbinding of CTLA4 to CTLA4-aCTLA4 on Treg surface in Tumor');
% R = addreaction(model, 'V_T.Treg_CTLA4_aCTLA4 + V_T.Mac_FcRIIIa <-> V_T.Treg_CTLA4_aCTLA4_FcRIIIa');
%     set (R, 'ReactionRate', 'kon_IgG1_FcR*(V_T.Treg_CTLA4_aCTLA4 * V_T.Mac_FcRIIIa)/A_Tcell -  koff_CTLA4_aCTLA4*V_T.Treg_CTLA4_aCTLA4_FcRIIIa');
%     set (R, 'Notes'       , 'binding and unbinding of CTLA4-aCTLA4 on Treg surface to Fc receptor on macrophages in Tumor');
% R = addreaction(model, 'V_T.Treg_CTLA4_aCTLA4_CTLA4 + V_T.Mac_FcRIIIa <-> V_T.Treg_CTLA4_aCTLA4_CTLA4_FcRIIIa');
%     set (R, 'ReactionRate', 'kon_IgG1_FcR*(V_T.Treg_CTLA4_aCTLA4_CTLA4 * V_T.Mac_FcRIIIa)/A_Tcell -  koff_CTLA4_aCTLA4*V_T.Treg_CTLA4_aCTLA4_CTLA4_FcRIIIa');
%     set (R, 'Notes'       , 'binding and unbinding of CTLA4-aCTLA4-CTLA4 on Treg surface to Fc receptor on macrophages in Tumor');

R = addreaction(model, 'V_P.Treg_CTLA4 <-> V_P.Treg_CTLA4_aCTLA4');
    set (R, 'ReactionRate', 'kon_CTLA4_aCTLA4*(V_P.Treg_CTLA4 * V_P.aCTLA4/gamma_P_aCTLA4) -  koff_CTLA4_aCTLA4*V_P.Treg_CTLA4_aCTLA4');
    set (R, 'Notes'       , 'binding and unbinding of CTLA4 to aCTLA4 on Treg surface in Peripheral compartment');
R = addreaction(model, 'V_P.Treg_CTLA4_aCTLA4 + V_P.Treg_CTLA4 <-> V_P.Treg_CTLA4_aCTLA4_CTLA4');
    set (R, 'ReactionRate', 'Chi_CTLA4_aCTLA4*kon_CTLA4_aCTLA4*(V_P.Treg_CTLA4 * V_P.Treg_CTLA4_aCTLA4)/A_Tcell -  koff_CTLA4_aCTLA4*V_P.Treg_CTLA4_aCTLA4_CTLA4');
    set (R, 'Notes'       , 'binding and unbinding of CTLA4 to CTLA4-aCTLA4 on Treg surface in Peripheral compartment');

% Treg Death through CTLA4 binding
R = addreaction(model,'V_T.Treg -> null');
    set(R,'ReactionRate','k_CTLA4_ADCC*V_T.Treg*H_Treg_T');
    set(R,'Notes','CTLA4 ADCC in the tumor compartment');
R = addreaction(model,'V_P.Treg -> null');
    set(R,'ReactionRate','k_CTLA4_ADCC*V_P.Treg*H_Treg_P');
    set(R,'Notes','CTLA4 ADCC in the peripheral compartment');
