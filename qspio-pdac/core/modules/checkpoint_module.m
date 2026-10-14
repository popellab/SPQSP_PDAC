% Checkpoint Module
%
% Models PD1 and CTLA4 Interactions
%
% Inputs: model       -- simbio model object with four compartments
%         params      -- object containing the default parameters
%         Tname       -- name of the T cell forming the checkpoint synapse
%         Cname       -- name of the cancer or APC cell forming the
%                        checkpoint synapse
%
% Outputs: model -- simbio model object with new PD1 module
%
% Note: This only works for Ti (i >= 1) interaction with APC and Cj. We
% would need to add a check for Tregs if we would like to generalize it to checkpoint
% model of Treg.
%
% KNOWN LIMITATION — Treg CTLA-4 trans-endocytosis (mechanism-audit consensus #8,
% deferred 2026-06-04). This synapse builds a fully DYNAMIC CD28-CD80/CD86 occupancy
% Hill (H_CD28_<Cname>, e.g. H_CD28_APC; SBML assignmentRule, NOT the frozen 0.1 in
% simbio_init.m — that is only the initial value) and models the priming T cell's OWN
% CTLA-4 competing with CD28 for the APC's B7 (CTLA4_CD80, PDL1_CD80_CTLA4 reactions).
% What it does NOT model is Treg-EXTRINSIC trans-endocytosis: Tregs express the highest
% CTLA-4 and physically strip CD80/86 off DCs (Qureshi 2011 Science PMID 21551058),
% depleting the B7 pool available to prime CD8 — a separate cell (Treg) acting on the
% shared DC supply, not the CD8's own checkpoint. The faithful insertion point would be
% the DC B7 supply (APC_CD80_total / APC_CD86_total, molecules per DC, set on the
% synapse via initialAssignment at lines ~272-273): a V_LN.Treg-dependent depletion sink
% there would flow through this existing synapse to lower CD28 occupancy naturally — NOT
% a Hill multiplier on H_CD28_APC (that would collide with the assignmentRule above).
% Deferred as a refinement (Med severity): the costim axis is already dynamic and
% checkpoint-aware, and the change is invasive to this binding model. See
% notes/model/mechanism_audit_reconciliation.md (#8) for the full adjudication.


function model = checkpoint_module(model,params,Tname,Cname)

% select the right compartment based on cancer or APC
if strcmpi(Cname(1),'C') || strcmpi(Cname,'C1') || strcmpi(Cname,'C2')
    compDrug = model.Compartment(3);
    gamma = 'gamma_T';
    Param_name = 'C';
    A_pre = 'A_cell';      % presenting-cell surface area (cancer cell)
    IFNg_src = 'V_T.IFNg'; % IFNg milieu driving PDL1 upregulation: tumor (cancer synapse)
elseif Cname(1)=='A'
    compDrug = model.Compartment(4);
    gamma = 'gamma_LN';
    Param_name = 'APC';
    IFNg_src = 'V_LN.IFNg';% IFNg milieu (mechanism #9, m1-A23): the APC priming synapse
                        % sits in the LN, so its IFNg-driven PDL1 upregulation must read the
                        % LN-local IFNg (V_LN.IFNg, added in cytokine_tcell_module.m), NOT the
                        % tumor IFNg. The previous V_T.IFNg was a cross-compartment leak that
                        % drove LN-APC PDL1 off tumor inflammation.
    A_pre = 'A_APC';    % presenting-cell surface area (APC); mechanism #4 (m1-A13):
                        % the PDL1/PDL2 secretion ceiling + internalization baseline below
                        % must use the APC's own area to convert molecules/cell -> density,
                        % matching the A_APC-based initialAssignment (lines ~285-291). They
                        % previously hardcoded A_cell, leaking cancer-cell geometry into the
                        % APC synapse (numerically ~1% today since A_cell~A_APC, but A_cell is
                        % derived from D_cell and would drift if cancer-cell diameter changes).
else
    error('Unknown cell type: %s. Expected cancer cell (C*, C1, C2) or APC.', Cname);
end

% Add the synapse compartment
comp = addcompartment(model,['syn_',Tname,'_',Cname],params.A_syn.Value,'CapacityUnits',params.A_syn.Units);
    set(comp,'Notes',['synapse compartment between ',Tname,' and ',Cname]);
    comp.UserData.Source = params.A_syn.Notes;
    comp.UserData.derivedFrom = params.A_syn.derivedFrom;

% Determine if checkpoint sizes have been defined before in antigen module
first_call = true;

try % see if synapse exist
    p = addparameter(model,'A_syn' ,params.A_syn.Value ,'ValueUnits',params.A_syn.Units, 'Notes', 'Surface area of the synapse ');
    p.UserData.Source = params.A_syn.Notes;
    p.UserData.derivedFrom = params.A_syn.derivedFrom;
catch
    first_call = false;
end
if first_call
% Add surface areas
p = addparameter(model,'A_Tcell' ,params.A_Tcell.Value ,'ValueUnits',params.A_Tcell.Units, 'Notes', 'Surface area of the T cell ');
p.UserData.Source = params.A_Tcell.Notes;
    p.UserData.derivedFrom = params.A_Tcell.derivedFrom;
p = addparameter(model,'A_cell' ,params.A_cell.Value ,'ValueUnits',params.A_cell.Units, 'Notes', 'Surface area of the Cancer cell ');
p.UserData.Source = params.A_cell.Notes;
    p.UserData.derivedFrom = params.A_cell.derivedFrom;
p = addparameter(model,'A_APC' ,params.A_APC.Value ,'ValueUnits',params.A_APC.Units, 'Notes', 'Surface area of the APC ');
p.UserData.Source = params.A_APC.Notes;
    p.UserData.derivedFrom = params.A_APC.derivedFrom;
p = addparameter(model,'d_syn' ,params.d_syn.Value ,'ValueUnits',params.d_syn.Units, 'Notes', 'The synapse gap distance (kd2D = kd3D*d_syn) ');
p.UserData.Source = params.d_syn.Notes;
    p.UserData.derivedFrom = params.d_syn.derivedFrom;
end

% Determine if first call
first_call = true;
try % see if synapse exist
kon = addparameter(model,'kon_PD1_PDL1',params.kon_PD1_PDL1.Value,'ValueUnits',params.kon_PD1_PDL1.Units, 'Notes', 'kon of PD1-PDL1 binding ');
kon.UserData.Source = params.kon_PD1_PDL1.Notes;
    kon.UserData.derivedFrom = params.kon_PD1_PDL1.derivedFrom;
catch
    first_call = false;
end

if first_call

k_out_PDL1 = addparameter(model,'k_out_PDL1',params.k_out_PDL1.Value,'ValueUnits',params.k_out_PDL1.Units, 'Notes', 'Expression rate of PDL1 on tumor cells');
k_out_PDL1.UserData.Source = params.k_out_PDL1.Notes;
    k_out_PDL1.UserData.derivedFrom = params.k_out_PDL1.derivedFrom;
k_in_PDL1 = addparameter(model,'k_in_PDL1',params.k_in_PDL1.Value,'ValueUnits',params.k_in_PDL1.Units, 'Notes', 'Degradation rate of PDL1 on tumor cells');
k_in_PDL1.UserData.Source = params.k_in_PDL1.Notes;
    k_in_PDL1.UserData.derivedFrom = params.k_in_PDL1.derivedFrom;
r_PDL1_IFNg = addparameter(model,'r_PDL1_IFNg',params.r_PDL1_IFNg.Value,'ValueUnits',params.r_PDL1_IFNg.Units, 'Notes', 'Number of folds increase of PDL1 expression by IFNg');
r_PDL1_IFNg.UserData.Source = params.r_PDL1_IFNg.Notes;
    r_PDL1_IFNg.UserData.derivedFrom = params.r_PDL1_IFNg.derivedFrom;

% Add kon Values
kon = addparameter(model,'kon_PD1_PDL2',params.kon_PD1_PDL2.Value,'ValueUnits',params.kon_PD1_PDL2.Units, 'Notes', 'kon of PD1-PDL2 binding ');
kon.UserData.Source = params.kon_PD1_PDL2.Notes;
    kon.UserData.derivedFrom = params.kon_PD1_PDL2.derivedFrom;
kon = addparameter(model,'kon_PD1_aPD1',params.kon_PD1_aPD1.Value,'ValueUnits',params.kon_PD1_aPD1.Units, 'Notes', 'kon of PD1-aPD1 binding ');
kon.UserData.Source = params.kon_PD1_aPD1.Notes;
    kon.UserData.derivedFrom = params.kon_PD1_aPD1.derivedFrom;
kon = addparameter(model,'kon_CD28_CD80',params.kon_CD28_CD80.Value,'ValueUnits',params.kon_CD28_CD80.Units, 'Notes', 'kon of CD28-CD80 binding ');
kon.UserData.Source = params.kon_CD28_CD80.Notes;
    kon.UserData.derivedFrom = params.kon_CD28_CD80.derivedFrom;
kon = addparameter(model,'kon_CD28_CD86',params.kon_CD28_CD86.Value,'ValueUnits',params.kon_CD28_CD86.Units, 'Notes', 'kon of CD28-CD86 binding ');
kon.UserData.Source = params.kon_CD28_CD86.Notes;
    kon.UserData.derivedFrom = params.kon_CD28_CD86.derivedFrom;
kon = addparameter(model,'kon_CTLA4_CD80',params.kon_CTLA4_CD80.Value,'ValueUnits',params.kon_CTLA4_CD80.Units, 'Notes', 'kon of CTLA4-CD80 binding ');
kon.UserData.Source = params.kon_CTLA4_CD80.Notes;
    kon.UserData.derivedFrom = params.kon_CTLA4_CD80.derivedFrom;
kon = addparameter(model,'kon_CTLA4_CD86',params.kon_CTLA4_CD86.Value,'ValueUnits',params.kon_CTLA4_CD86.Units, 'Notes', 'kon of CTLA4-CD86 binding ');
kon.UserData.Source = params.kon_CTLA4_CD86.Notes;
    kon.UserData.derivedFrom = params.kon_CTLA4_CD86.derivedFrom;
kon = addparameter(model,'kon_CD80_PDL1',params.kon_CD80_PDL1.Value,'ValueUnits',params.kon_CD80_PDL1.Units, 'Notes', 'kon of CD80-PDL1 binding ');
kon.UserData.Source = params.kon_CD80_PDL1.Notes;
    kon.UserData.derivedFrom = params.kon_CD80_PDL1.derivedFrom;
kon = addparameter(model,'kon_CD80_CD80',params.kon_CD80_CD80.Value,'ValueUnits',params.kon_CD80_CD80.Units, 'Notes', 'kon of CD80 self-association ');
kon.UserData.Source = params.kon_CD80_CD80.Notes;
    kon.UserData.derivedFrom = params.kon_CD80_CD80.derivedFrom;

% Add koff Values
koff = addparameter(model,'koff_PD1_PDL1' ,params.koff_PD1_PDL1.Value ,'ValueUnits',params.koff_PD1_PDL1.Units, 'Notes', 'koff of PD1-PDL1 binding ');
koff.UserData.Source = params.koff_PD1_PDL1.Notes;
    koff.UserData.derivedFrom = params.koff_PD1_PDL1.derivedFrom;
koff = addparameter(model,'koff_PD1_PDL2' ,params.koff_PD1_PDL2.Value ,'ValueUnits',params.koff_PD1_PDL2.Units, 'Notes', 'koff of PD1-PDL2 binding ');
koff.UserData.Source = params.koff_PD1_PDL2.Notes;
    koff.UserData.derivedFrom = params.koff_PD1_PDL2.derivedFrom;
koff = addparameter(model,'koff_PD1_aPD1' ,params.koff_PD1_aPD1.Value ,'ValueUnits',params.koff_PD1_aPD1.Units, 'Notes', 'koff of PD1-aPD1 binding ');
koff.UserData.Source = params.koff_PD1_aPD1.Notes;
    koff.UserData.derivedFrom = params.koff_PD1_aPD1.derivedFrom;
koff = addparameter(model,'koff_CD28_CD80' ,params.koff_CD28_CD80.Value ,'ValueUnits',params.koff_CD28_CD80.Units, 'Notes', 'koff of CD28-CD80 binding ');
koff.UserData.Source = params.koff_CD28_CD80.Notes;
    koff.UserData.derivedFrom = params.koff_CD28_CD80.derivedFrom;
koff = addparameter(model,'koff_CD28_CD86' ,params.koff_CD28_CD86.Value ,'ValueUnits',params.koff_CD28_CD86.Units, 'Notes', 'koff of CD28-CD86 binding ');
koff.UserData.Source = params.koff_CD28_CD86.Notes;
    koff.UserData.derivedFrom = params.koff_CD28_CD86.derivedFrom;
koff = addparameter(model,'koff_CTLA4_CD80' ,params.koff_CTLA4_CD80.Value ,'ValueUnits',params.koff_CTLA4_CD80.Units, 'Notes', 'koff of CTLA4-CD80 binding ');
koff.UserData.Source = params.koff_CTLA4_CD80.Notes;
    koff.UserData.derivedFrom = params.koff_CTLA4_CD80.derivedFrom;
koff = addparameter(model,'koff_CTLA4_CD86' ,params.koff_CTLA4_CD86.Value ,'ValueUnits',params.koff_CTLA4_CD86.Units, 'Notes', 'koff of CTLA4-CD86 binding ');
koff.UserData.Source = params.koff_CTLA4_CD86.Notes;
    koff.UserData.derivedFrom = params.koff_CTLA4_CD86.derivedFrom;
koff = addparameter(model,'koff_CD80_PDL1' ,params.koff_CD80_PDL1.Value ,'ValueUnits',params.koff_CD80_PDL1.Units, 'Notes', 'koff of CD80-PDL1 binding ');
koff.UserData.Source = params.koff_CD80_PDL1.Notes;
    koff.UserData.derivedFrom = params.koff_CD80_PDL1.derivedFrom;
koff = addparameter(model,'koff_CD80_CD80',params.koff_CD80_CD80.Value,'ValueUnits',params.koff_CD80_CD80.Units, 'Notes', 'koff of CD80 self-association ');
koff.UserData.Source = params.koff_CD80_CD80.Notes;
    koff.UserData.derivedFrom = params.koff_CD80_CD80.derivedFrom;

% Bivalent anibody parameters
p = addparameter(model,'Chi_PD1_aPD1' ,params.Chi_PD1_aPD1.Value,'ValueUnits',params.Chi_PD1_aPD1.Units, 'Notes', 'Antibody cross-arm binding efficiency that also includes the conversion of kon from 3D to 2D ');
p.UserData.Source = params.Chi_PD1_aPD1.Notes;
    p.UserData.derivedFrom = params.Chi_PD1_aPD1.derivedFrom;

% PD1-related Hill parameters
p = addparameter(model,'PD1_50',params.PD1_50.Value,'ValueUnits',params.PD1_50.Units, 'Notes', 'PD1/PDL1 concentration for half-maximal T cell inactivation ');
p.UserData.Source = params.PD1_50.Notes;
    p.UserData.derivedFrom = params.PD1_50.derivedFrom;
p = addparameter(model,'n_PD1',params.n_PD1.Value,'ValueUnits',params.n_PD1.Units, 'Notes', 'Hill coefficient for PD1/PDL1 half-maximal T cell inactivation ');
p.UserData.Source = params.n_PD1.Notes;
    p.UserData.derivedFrom = params.n_PD1.derivedFrom;
p = addparameter(model,'CD28_CD8X_50',params.CD28_CD8X_50.Value,'ValueUnits',params.CD28_CD8X_50.Units, 'Notes', 'CD28-CD80/CD28-CD86 concentration for half-maximal T cell co-estimulation ');
p.UserData.Source = params.CD28_CD8X_50.Notes;
    p.UserData.derivedFrom = params.CD28_CD8X_50.derivedFrom;
p = addparameter(model,'n_CD28_CD8X',params.n_CD28_CD8X.Value,'ValueUnits',params.n_CD28_CD8X.Units, 'Notes', 'Hill coefficient for CD28-CD80/CD28-CD86 half-maximal T cell co-estimulation ');
p.UserData.Source = params.n_CD28_CD8X.Notes;
    p.UserData.derivedFrom = params.n_CD28_CD8X.derivedFrom;

end

% Checkpoint Expressions
% Check if T cell has defined before
first_Tcell_call = true;
try
    p = addparameter(model,[Tname,'_PD1_total'],params.T8_PD1.Value,'ValueUnits',params.T8_PD1.Units,'ConstantValue',false, 'Notes', ['concentration of PD1 on ',Tname,' cells ']);
    p.UserData.Source = params.T8_PD1.Notes;
catch
    first_Tcell_call = false;
end
if first_Tcell_call
        p.UserData.derivedFrom = params.T8_PD1.derivedFrom;
    p = addparameter(model,[Tname,'_CD28_total'],params.T8_CD28.Value,'ValueUnits',params.T8_CD28.Units,'ConstantValue',false, 'Notes', ['concentration of CD28 on ',Tname,' cells ']);
    p.UserData.Source = params.T8_CD28.Notes;
        p.UserData.derivedFrom = params.T8_CD28.derivedFrom;
    p = addparameter(model,[Tname,'_CTLA4_syn'],params.T8_CTLA4.Value,'ValueUnits',params.T8_CTLA4.Units,'ConstantValue',false, 'Notes', ['concentration of CTLA4 on ',Tname,' cells ']);
    p.UserData.Source = params.T8_CTLA4.Notes;
        p.UserData.derivedFrom = params.T8_CTLA4.derivedFrom;
    p = addparameter(model,[Tname,'_PDL1_total'],params.T8_PDL1.Value,'ValueUnits',params.T8_PDL1.Units,'ConstantValue',false, 'Notes', ['concentration of PDL1 on ',Tname,' cells ']);
    p.UserData.Source = params.T8_PDL1.Notes;
        p.UserData.derivedFrom = params.T8_PDL1.derivedFrom;
end
% Check if Cancer or APC has defined before
first_Ccell_call = true;
try
    p = addparameter(model,[Cname,'_PDL1_base'],params.([Param_name,'_PDL1']).Value,'ValueUnits',params.([Param_name,'_PDL1']).Units,'ConstantValue',false, 'Notes', ['baseline number of PDL1 molecules per ',Cname,' cell ' params.([Param_name,'_PDL1']).Notes]);
catch
    first_Ccell_call = false;
end
if first_Ccell_call
    p = addparameter(model,['r_PDL2',Cname],params.(['r_PDL2',Param_name]).Value,'ValueUnits',params.(['r_PDL2',Param_name]).Units,'ConstantValue',false, 'Notes', ['PDL2/PDL1 molecule ratio on ', Cname,'cell ', params.(['r_PDL2',Param_name]).Notes]);

    p = addparameter(model,[Cname,'_CD80_total'],params.([Param_name,'_CD80']).Value,'ValueUnits',params.([Param_name,'_CD80']).Units,'ConstantValue',false, 'Notes', ['number of CD80 molecules per ',Cname,' cell ' params.([Param_name,'_CD80']).Notes]);
    p = addparameter(model,[Cname,'_CD86_total'],params.([Param_name,'_CD86']).Value,'ValueUnits',params.([Param_name,'_CD86']).Units,'ConstantValue',false, 'Notes', ['number of CD86 molecules per ',Cname,' cell ' params.([Param_name,'_CD86']).Notes]);
end

% Add Species
x = addparameter(model, [comp.Name,'_PDL1_total'],0,'ValueUnits','molecule/micrometer^2','ConstantValue',false, 'Notes', 'concentration of bound and unbound PDL1 molecules ');
x = addparameter(model, [comp.Name,'_PDL2_total'],0,'ValueUnits','molecule/micrometer^2','ConstantValue',false, 'Notes', 'concentration of bound and unbound PDL2 molecules ');

x = addspecies(comp,'PD1_PDL1',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of PD1-PDL1 complex');
x = addspecies(comp,'PD1_PDL2',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of PD1-PDL2 complex');
x = addspecies(comp,'PD1',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of PD1 in synapse');
x = addspecies(comp,'PDL1',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of PDL1 in synapse');
x = addspecies(comp,'PDL2',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of PDL2 in synapse');
x = addspecies(comp,'PD1_aPD1',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of PD1-aPD1 complex');
x = addspecies(comp,'PD1_aPD1_PD1',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of PD1-aPD1-PD1 complex');
x = addspecies(comp,'TPDL1',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of PDL1 in synapse of T cell');

x = addspecies(comp,'CD28_CD80',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of CD28-CD80 complex');
x = addspecies(comp,'CD28_CD80_CD28',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of CD28-CD80-CD28 complex');
x = addspecies(comp,'CD28_CD86',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of CD28-CD86 complex');
x = addspecies(comp,'CD80_CTLA4',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of CD80-CTLA4 complex');
x = addspecies(comp,'CD80_CTLA4_CD80',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of CD80-CTLA4-CD80 complex');
x = addspecies(comp,'CTLA4_CD80_CTLA4',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of CTLA4-CD80-CTLA4 complex');
x = addspecies(comp,'CD80_CTLA4_CD80_CTLA4',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of CD80-CTLA4-CD80-CTLA4 complex');
x = addspecies(comp,'CD86_CTLA4',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of CD86-CTLA4 complex');
x = addspecies(comp,'CD86_CTLA4_CD86',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of CD86-CTLA4-CD86 complex');
x = addspecies(comp,'PDL1_CD80',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of PDL1-CD80 complex');
x = addspecies(comp,'PDL1_CD80_CD28',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of PDL1-CD80-CD28 complex');
x = addspecies(comp,'PDL1_CD80_CTLA4',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of PDL1-CD80-CTLA4 complex');
x = addspecies(comp,'CD28',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of CD28 in synapse');
x = addspecies(comp,'CTLA4',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of CTLA4 in synapse');
x = addspecies(comp,'CD80',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of CD80 dimer in synapse');
x = addspecies(comp,'CD80m',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of CD80 monomer in synapse');
x = addspecies(comp,'CD86',0,'InitialAmountUnits','molecule/micrometer^2', 'Notes', 'concentration of CD86 in synapse');

% Update Input Parameters
addrule(model,[comp.Name,'.PD1',' = '  ,Tname,'_PD1_total /A_Tcell' ] ,'initialAssignment');
addrule(model,[comp.Name,'.CD28',' = ' ,Tname,'_CD28_total /A_Tcell'] ,'initialAssignment');
addrule(model,[comp.Name,'.CTLA4',' = ',Tname,'_CTLA4_syn /A_Tcell'   ] ,'initialAssignment');
addrule(model,[comp.Name,'.TPDL1',' = ',Tname,'_PDL1_total /A_Tcell'] ,'initialAssignment');
if Cname(1)=='C'
    addrule(model,[comp.Name,'.PDL1 = ',Cname,'_PDL1_base /A_cell'] ,'initialAssignment');
    addrule(model,[comp.Name,'.PDL2 = ',Cname,'_PDL1_base*r_PDL2',Cname,' /A_cell'] ,'initialAssignment');
    addrule(model,[comp.Name,'_PDL1_total = ',comp.Name,'.PDL1+',comp.Name,'.PD1_PDL1+',comp.Name,'.PDL1_CD80+',...
                                              comp.Name,'.PDL1_CD80_CD28+',comp.Name,'.PDL1_CD80_CTLA4'] ,'repeatedAssignment');
    addrule(model,[comp.Name,'_PDL2_total = ',comp.Name,'.PD1_PDL2+',comp.Name,'.PDL2'] ,'repeatedAssignment');
    addrule(model,[comp.Name,'.CD80 = ',Cname,'_CD80_total /A_cell'] ,'initialAssignment');
    addrule(model,[comp.Name,'.CD86 = ',Cname,'_CD86_total /A_cell'] ,'initialAssignment');
elseif Cname(1)=='A'
    addrule(model,[comp.Name,'.PDL1 = ',Cname,'_PDL1_base /A_APC'] ,'initialAssignment');
    addrule(model,[comp.Name,'.PDL2 = ',Cname,'_PDL1_base*r_PDL2',Cname,' /A_APC'] ,'initialAssignment');
    addrule(model,[comp.Name,'_PDL1_total = ',comp.Name,'.PDL1+',comp.Name,'.PD1_PDL1+',comp.Name,'.PDL1_CD80+',...
                                              comp.Name,'.PDL1_CD80_CD28+',comp.Name,'.PDL1_CD80_CTLA4'] ,'repeatedAssignment');
    addrule(model,[comp.Name,'_PDL2_total = ',comp.Name,'.PD1_PDL2+',comp.Name,'.PDL2'] ,'repeatedAssignment');
    addrule(model,[comp.Name,'.CD80 = ',Cname,'_CD80_total /A_APC'] ,'initialAssignment');
    addrule(model,[comp.Name,'.CD86 = ',Cname,'_CD86_total /A_APC'] ,'initialAssignment');
end

% PDL1 Secretion
%
% IFNg sets the CEILING, not the rate (fixed 2026-07-13).
%
% The previous form was
%     k_out_PDL1 * H_IFNg * (1 - PDL1_total/(base*r_PDL1_IFNg/A_pre))
% i.e. the IFNg Hill multiplied the RATE while the ceiling sat at base*r_PDL1_IFNg
% regardless of IFNg. Paired with the internalization reaction below (which pulls
% toward `base`), the steady state solves to k_out/k_in ~ 1330 -- so synthesis
% overwhelms internalization even at a tiny Hill, and PD-L1 climbs to ~74% of its
% MAXIMAL 6x induction in a cold, untreated tumor whose IFNg (0.16 pM) is 18x BELOW
% the induction EC50 (2.96 pM). IFNg only set how FAST it got there, not how high.
%
% Consequence: H_PD1_C1 = 0.92 at untreated baseline instead of the ~0.44 this
% parameterization was designed for (see the sanity check in parameters_PDAC.m,
% which explicitly warns that PD-1 must be "a meaningful dynamic lever instead of a
% saturated constant"). PD-1 became the dominant brake on CD8 killing, so anti-PD-1
% released a 17x killing boost -- irreconcilable with anti-PD-1 monotherapy's ~0% ORR
% in PDAC.
%
% Now the ceiling itself scales with IFNg: base at H=0, base*r_PDL1_IFNg at H=1. A
% cold tumor rests at base PD-L1 (H_PD1_C1 ~ 0.44); an inflamed one climbs toward 6x.
% The term goes negative above the ceiling, so PD-L1 also falls back when IFNg drops
% -- the same sign convention the internalization reaction already uses.
% The IFNg Hill is kept on the RATE as well as the ceiling. Dropping it from the rate
% (an earlier attempt) makes baseline synthesis ~19x faster, stiffens the system, and
% breaks MATLAB<->C++ parity in the near-zero inflammatory-signal regime. Keeping it
% leaves the rate magnitude untouched -- the fix rides entirely in the ceiling.
R = addreaction(model,['null -> ',comp.Name,'.PDL1']);
    set (R, 'ReactionRate', ['k_out_PDL1*',IFNg_src,'/(',IFNg_src,'+IFNg_50_ind)*(1-',...
    comp.Name,'_PDL1_total/(',Cname,'_PDL1_base*(1+(r_PDL1_IFNg-1)*',...
    IFNg_src,'/(',IFNg_src,'+IFNg_50_ind))/',A_pre,'))']);
    set (R, 'Notes'       , 'Translocation of PDL1 between cell surface and cytoplasm; IFNg sets the induction CEILING (base -> base*r_PDL1_IFNg), not just the rate');
R = addreaction(model,['null -> ',comp.Name,'.PDL2']);
    set (R, 'ReactionRate', ['k_out_PDL1*r_PDL2',Cname,'*',IFNg_src,'/(',IFNg_src,'+IFNg_50_ind)*(1-',...
    comp.Name,'_PDL2_total/(',Cname,'_PDL1_base*(1+(r_PDL1_IFNg-1)*',...
    IFNg_src,'/(',IFNg_src,'+IFNg_50_ind))/',A_pre,'*r_PDL2',Cname,'))']);
    set (R, 'Notes'       , 'Translocation of PDL2 between cell surface and cytoplasm; IFNg sets the induction CEILING, not just the rate');
R = addreaction(model,['null -> ' comp.Name,'.PDL1']);
    set (R, 'ReactionRate', ['k_in_PDL1*(',Cname,'_PDL1_base/',A_pre,'-',comp.Name,'_PDL1_total)']);
    set (R, 'Notes'       , 'Translocation of PDL1 between cell surface and cytoplasm');
R = addreaction(model,['null -> ' comp.Name,'.PDL2']);
    set (R, 'ReactionRate', ['k_in_PDL1*(',Cname,'_PDL1_base/',A_pre,'*r_PDL2',Cname,'-',comp.Name,'_PDL2_total)']);
    set (R, 'Notes'       , 'Translocation of PDL2 between cell surface and cytoplasm');

% Dynamics of PD1/PDL1/PDL2/aPD1
R = addreaction(model,[comp.Name,'.PD1 + ',comp.Name,'.PDL1 <-> ',comp.Name,'.PD1_PDL1']);
    set (R, 'ReactionRate', ['kon_PD1_PDL1*(',comp.Name,'.PD1)*(',comp.Name,'.PDL1)  -  koff_PD1_PDL1*',comp.Name,'.PD1_PDL1']);
    set (R, 'Notes'       , 'binding and unbinding of PD1 PDL1 in synapse');
R = addreaction(model,[comp.Name,'.PD1 + ',comp.Name,'.PDL2 <-> ',comp.Name,'.PD1_PDL2']);
    set (R, 'ReactionRate', ['kon_PD1_PDL2*(',comp.Name,'.PD1)*(',comp.Name,'.PDL2)  -  koff_PD1_PDL2*',comp.Name,'.PD1_PDL2']);
    set (R, 'Notes'       , 'binding and unbinding of PD1 PDL2 in synapse');
R = addreaction(model,[comp.Name,'.PD1 <-> ',comp.Name,'.PD1_aPD1']);
    set (R, 'ReactionRate', ['2*kon_PD1_aPD1*(',comp.Name,'.PD1 * ',compDrug.Name,'.aPD1/',gamma,'_aPD1) -  koff_PD1_aPD1*',comp.Name,'.PD1_aPD1']);
    set (R, 'Notes'       , ['binding and unbinding of PD1 to aPD1 on ',Tname,' surface in synapse']);
R = addreaction(model,[comp.Name,'.PD1_aPD1 + ',comp.Name,'.PD1 <-> ',comp.Name,'.PD1_aPD1_PD1']);
    set (R, 'ReactionRate', ['Chi_PD1_aPD1*kon_PD1_aPD1*(',comp.Name,'.PD1 * ',comp.Name,'.PD1_aPD1) -  2*koff_PD1_aPD1*',comp.Name,'.PD1_aPD1_PD1']);
    set (R, 'Notes'       , ['binding and unbinding of PD1 to PD1-aPD1 on ',Tname,' surface in synapse']);
% Dynamics of CD28/CTLA4/CD80/CD86
% CD28-CD80
R = addreaction(model,[comp.Name,'.CD28 + ',comp.Name,'.CD80 <-> ',comp.Name,'.CD28_CD80']);
    set (R, 'ReactionRate', ['2*kon_CD28_CD80*(',comp.Name,'.CD28)*(',comp.Name,'.CD80)  -  koff_CD28_CD80*',comp.Name,'.CD28_CD80']);
    set (R, 'Notes'       , 'binding and unbinding of CD28 and CD80 in synapse');
R = addreaction(model,[comp.Name,'.CD28_CD80 + ',comp.Name,'.CD28 <-> ',comp.Name,'.CD28_CD80_CD28']);
    set (R, 'ReactionRate', ['kon_CD28_CD80*(',comp.Name,'.CD28)*(',comp.Name,'.CD28_CD80)  -  2*koff_CD28_CD80*',comp.Name,'.CD28_CD80_CD28']);
    set (R, 'Notes'       , 'binding and unbinding of CD28-CD80 and CD28 in synapse');
% CD28-CD86
R = addreaction(model,[comp.Name,'.CD28 + ',comp.Name,'.CD86 <-> ',comp.Name,'.CD28_CD86']);
    set (R, 'ReactionRate', ['kon_CD28_CD86*(',comp.Name,'.CD28)*(',comp.Name,'.CD86)  -  koff_CD28_CD86*',comp.Name,'.CD28_CD86']);
    set (R, 'Notes'       , 'binding and unbinding of CD28 and CD86 in synapse');
% CTLA4-CD80
R = addreaction(model,[comp.Name,'.CTLA4 + ',comp.Name,'.CD80 <-> ',comp.Name,'.CD80_CTLA4']);
    set (R, 'ReactionRate', ['4*kon_CTLA4_CD80*(',comp.Name,'.CTLA4)*(',comp.Name,'.CD80)  -  koff_CTLA4_CD80*',comp.Name,'.CD80_CTLA4']);
    set (R, 'Notes'       , 'binding and unbinding of CTLA4 and CD80 in synapse');
R = addreaction(model,[comp.Name,'.CTLA4 + ',comp.Name,'.CD80_CTLA4 <-> ',comp.Name,'.CTLA4_CD80_CTLA4']);
    set (R, 'ReactionRate', ['2*kon_CTLA4_CD80*(',comp.Name,'.CTLA4)*(',comp.Name,'.CD80_CTLA4)  -  2*koff_CTLA4_CD80*',comp.Name,'.CTLA4_CD80_CTLA4']);
    set (R, 'Notes'       , 'binding and unbinding of CTLA4 and CD80-CTLA4 in synapse');
R = addreaction(model,[comp.Name,'.CD80 + ',comp.Name,'.CTLA4_CD80_CTLA4 <-> ',comp.Name,'.CD80_CTLA4_CD80_CTLA4']);
    set (R, 'ReactionRate', ['4*kon_CTLA4_CD80*(',comp.Name,'.CD80)*(',comp.Name,'.CTLA4_CD80_CTLA4)  -  koff_CTLA4_CD80*',comp.Name,'.CD80_CTLA4_CD80_CTLA4']);
    set (R, 'Notes'       , 'binding and unbinding of CD80 and CTLA4-CD80-CTLA4 in synapse');
R = addreaction(model,[comp.Name,'.CD80_CTLA4 + ',comp.Name,'.CD80 <-> ',comp.Name,'.CD80_CTLA4_CD80']);
    set (R, 'ReactionRate', ['2*kon_CTLA4_CD80*(',comp.Name,'.CD80_CTLA4)*(',comp.Name,'.CD80)  -  2*koff_CTLA4_CD80*',comp.Name,'.CD80_CTLA4_CD80']);
    set (R, 'Notes'       , 'binding and unbinding of CD80-CTLA4 and CD80 in synapse');
R = addreaction(model,[comp.Name,'.CTLA4 + ',comp.Name,'.CD80_CTLA4_CD80 <-> ',comp.Name,'.CD80_CTLA4_CD80_CTLA4']);
    set (R, 'ReactionRate', ['4*kon_CTLA4_CD80*(',comp.Name,'.CTLA4)*(',comp.Name,'.CD80_CTLA4_CD80)  -  koff_CTLA4_CD80*',comp.Name,'.CD80_CTLA4_CD80_CTLA4']);
    set (R, 'Notes'       , 'binding and unbinding of CTLA4 and CD80-CTLA4-CD80 in synapse');
% CTLA4-CD86
R = addreaction(model,[comp.Name,'.CTLA4 + ',comp.Name,'.CD86 <-> ',comp.Name,'.CD86_CTLA4']);
    set (R, 'ReactionRate', ['2*kon_CTLA4_CD86*(',comp.Name,'.CTLA4)*(',comp.Name,'.CD86)  -  koff_CTLA4_CD86*',comp.Name,'.CD86_CTLA4']);
    set (R, 'Notes'       , 'binding and unbinding of CTLA4 and CD86 in synapse');
R = addreaction(model,[comp.Name,'.CD86_CTLA4 + ',comp.Name,'.CD86 <-> ',comp.Name,'.CD86_CTLA4_CD86']);
    set (R, 'ReactionRate', ['kon_CTLA4_CD86*(',comp.Name,'.CD86_CTLA4)*(',comp.Name,'.CD86)  -  2*koff_CTLA4_CD86*',comp.Name,'.CD86_CTLA4_CD86']);
    set (R, 'Notes'       , 'binding and unbinding of CD86-CTLA4 and CD86 in synapse');
% CD80 dimer dissociation
R = addreaction(model,[comp.Name,'.CD80m + ',comp.Name,'.CD80m <-> ',comp.Name,'.CD80']);
    set (R, 'ReactionRate', ['kon_CD80_CD80*(',comp.Name,'.CD80m)*(',comp.Name,'.CD80m) - koff_CD80_CD80*',comp.Name,'.CD80']);
    set (R, 'Notes'       , 'self-association and dissociation of CD80 monomers in synapse');
% cis PDL1-CD80-CD28
R = addreaction(model,[comp.Name,'.CD80m + ',comp.Name,'.PDL1 <-> ',comp.Name,'.PDL1_CD80']);
    set (R, 'ReactionRate', ['kon_CD80_PDL1*(',comp.Name,'.CD80m)*(',comp.Name,'.PDL1)  -  koff_CD80_PDL1*',comp.Name,'.PDL1_CD80']);
    set (R, 'Notes'       , 'binding and unbinding of CD80 and PDL1 in synapse');
R = addreaction(model,[comp.Name,'.PDL1_CD80 + ',comp.Name,'.CD28 <-> ',comp.Name,'.PDL1_CD80_CD28']);
    set (R, 'ReactionRate', ['kon_CD28_CD80*(',comp.Name,'.PDL1_CD80)*(',comp.Name,'.CD28)  - koff_CD28_CD80*',comp.Name,'.PDL1_CD80_CD28']);
    set (R, 'Notes'       , 'binding and unbinding of PDL1-CD80 and CD28 in synapse');
R = addreaction(model,[comp.Name,'.PDL1_CD80 + ',comp.Name,'.CTLA4 <-> ',comp.Name,'.PDL1_CD80_CTLA4']);
    set (R, 'ReactionRate', ['2*kon_CTLA4_CD80*(',comp.Name,'.PDL1_CD80)*(',comp.Name,'.CTLA4)  - koff_CTLA4_CD80*',comp.Name,'.PDL1_CD80_CTLA4']);
    set (R, 'Notes'       , 'binding and unbinding of PDL1-CD80 and CTLA4 in synapse');
% Update PD1 Hill Function
addrule(model,['H_PD1_',Cname,' = ((',comp.Name,'.PD1_PDL1+',comp.Name,'.PD1_PDL2)/PD1_50)^n_PD1/(((',comp.Name,'.PD1_PDL1+',comp.Name,'.PD1_PDL2)/PD1_50)^n_PD1 + 1)'],'repeatedAssignment');
% Anti-PD-1 occupancy fraction of synaptic PD1 (drug-engagement gate, =0 without
% aPD1, ->1 at saturating blockade; PD1_aPD1_PD1 is bivalent so counts 2x). Used by
% Teff_module to gate reinvigoration of exhausted CD8 (mid-tier cA-C4). Only the
% tumor CD8-cancer (C1) synapse drives reinvigoration.
if strcmpi(Cname,'C1')
    addparameter(model,'phi_aPD1_C1',0,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Fraction of synaptic PD1 sequestered by anti-PD-1 (drug-engagement gate for CD8 reinvigoration)');
    addrule(model,['phi_aPD1_C1 = (',comp.Name,'.PD1_aPD1 + 2*',comp.Name,'.PD1_aPD1_PD1) / (',comp.Name,'.PD1 + ',comp.Name,'.PD1_PDL1 + ',comp.Name,'.PD1_PDL2 + ',comp.Name,'.PD1_aPD1 + 2*',comp.Name,'.PD1_aPD1_PD1)'],'repeatedAssignment');
end
% NOTE (Path B caveat): this B7-CD28 synapse Hill sits saturated (~0.98) and
% treatment-invariant even at the help-poor cold-tumor baseline, because B7
% (CD80/CD86) is treated as constitutively high. Biologically B7 is UPREGULATED
% by CD40L licensing of the DC, so in a help-limited PDAC baseline the synapse
% should read LOWER, not saturated. The cleaner long-term form is to make this
% Hill's B7 licensing-dependent (scale CD80/CD86 by H_CD4_help). For now the CD8
% division-destiny term (cytokine_tcell_module N_aT) multiplies by H_CD4_help as
% a shortcut to the same effect — see notes/calibration/cd8_fc_vaccine_expansion_deficit.md.
addrule(model,['H_CD28_',Cname,' = ((',comp.Name,'.CD28_CD80 + ',comp.Name,'.CD28_CD86 + 2*',comp.Name,'.CD28_CD80_CD28 + ',comp.Name,'.PDL1_CD80_CD28)/CD28_CD8X_50)^n_CD28_CD8X/(((',comp.Name,'.CD28_CD80 + ',comp.Name,'.CD28_CD86 + 2*',comp.Name,'.CD28_CD80_CD28 + ',comp.Name,'.PDL1_CD80_CD28)/CD28_CD8X_50)^n_CD28_CD8X + 1)'],'repeatedAssignment');
