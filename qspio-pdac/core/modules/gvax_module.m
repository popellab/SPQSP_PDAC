function model = gvax_module(model, params, varargin)
% GVAX module with intradermal injection site compartment
% - GVAX cells administered to intradermal injection site (V_ID)
% - Local APC recruitment and maturation at injection site
% - Mature APCs migrate from injection site to lymph nodes
% - Optional: cyclophosphamide for Treg depletion
%
% Dosing is handled through schedule_dosing('GVAX') which doses into V_ID.GVAX_cells
%
% Optional parameters:
%   'add_cyclophosphamide' - logical flag to add Cy for Treg depletion (default: false)

% Parse optional inputs
p = inputParser;
addParameter(p, 'add_cyclophosphamide', false, @islogical);
parse(p, varargin{:});
add_cy = p.Results.add_cyclophosphamide;

%% ---- Add intradermal injection site compartment and species ----
% Create intradermal injection site compartment
V_ID = sbioselect(model, 'Name', 'V_ID', 'Type', 'compartment');
if isempty(V_ID)
    V_ID = addcompartment(model, 'V_ID', 'CapacityUnits', 'liter');
    set(V_ID, 'Notes', 'Intradermal injection site compartment');
end

% Add GVAX_cells species in injection site (target for dosing)
gvax_cells = sbioselect(model, 'Name', 'GVAX_cells', 'Type', 'species');
if isempty(gvax_cells)
    gvax_cells = addspecies(V_ID, 'GVAX_cells', 0, 'InitialAmountUnits', 'cell');
    set(gvax_cells, 'Notes', 'GVAX vaccine cells at intradermal injection site');
end

% Add APC species at injection site
apc_id = sbioselect(model, 'Name', 'APC', 'Type', 'species', 'Parent', V_ID);
if isempty(apc_id)
    apc_id = addspecies(V_ID, 'APC', 0, 'InitialAmountUnits', 'cell');
    set(apc_id, 'Notes', 'Naive APCs recruited to injection site');
end

% Add mature APC species at injection site
mapc_id = sbioselect(model, 'Name', 'mAPC', 'Type', 'species', 'Parent', V_ID);
if isempty(mapc_id)
    mapc_id = addspecies(V_ID, 'mAPC', 0, 'InitialAmountUnits', 'cell');
    set(mapc_id, 'Notes', 'Mature APCs with antigen at injection site');
end

% Add GM-CSF species at injection site
gmcsf_id = sbioselect(model, 'Name', 'GMCSF', 'Type', 'species', 'Parent', V_ID);
if isempty(gmcsf_id)
    gmcsf_id = addspecies(V_ID, 'GMCSF', 0, 'InitialAmountUnits', 'nanomole');
    set(gmcsf_id, 'Notes', 'GM-CSF amount at injection site');
end

% Add P1_GVAX antigen species at injection site
p1_gvax_id = sbioselect(model, 'Name', 'P1_GVAX', 'Type', 'species', 'Parent', V_ID);
if isempty(p1_gvax_id)
    p1_gvax_id = addspecies(V_ID, 'P1_GVAX', 0, 'InitialAmountUnits', 'nanomole');
    set(p1_gvax_id, 'Notes', 'GVAX-derived antigen at injection site');
end

% Load parameters from params structure

% GVAX cell clearance at injection site
k_GVAX = sbioselect(model,'Type','parameter','Name','k_GVAX_clearance');
if isempty(k_GVAX), k_GVAX = addparameter(model,'k_GVAX_clearance',params.k_GVAX_clearance.Value,'ValueUnits',params.k_GVAX_clearance.Units); end
set(k_GVAX, 'Notes', params.k_GVAX_clearance.Notes);

% Injection site volume
V_ID_vol = sbioselect(model,'Type','parameter','Name','V_ID_volume');
if isempty(V_ID_vol), V_ID_vol = addparameter(model,'V_ID_volume',params.V_ID_volume.Value,'ValueUnits',params.V_ID_volume.Units); end
set(V_ID_vol, 'Notes', params.V_ID_volume.Notes);
set(V_ID, 'Capacity', V_ID_vol.Value);

% GM-CSF production and degradation parameters
k_GMCSF_prod = sbioselect(model,'Type','parameter','Name','k_GMCSF_prod');
if isempty(k_GMCSF_prod), k_GMCSF_prod = addparameter(model,'k_GMCSF_prod',params.k_GMCSF_prod.Value,'ValueUnits',params.k_GMCSF_prod.Units); end
set(k_GMCSF_prod, 'Notes', params.k_GMCSF_prod.Notes);

k_GMCSF_deg = sbioselect(model,'Type','parameter','Name','k_GMCSF_deg');
if isempty(k_GMCSF_deg), k_GMCSF_deg = addparameter(model,'k_GMCSF_deg',params.k_GMCSF_deg.Value,'ValueUnits',params.k_GMCSF_deg.Units); end
set(k_GMCSF_deg, 'Notes', params.k_GMCSF_deg.Notes);

% APC recruitment to injection site
k_APC_recruit_ID = sbioselect(model,'Type','parameter','Name','k_APC_recruit_ID');
if isempty(k_APC_recruit_ID), k_APC_recruit_ID = addparameter(model,'k_APC_recruit_ID',params.k_APC_recruit_ID.Value,'ValueUnits',params.k_APC_recruit_ID.Units); end
set(k_APC_recruit_ID, 'Notes', params.k_APC_recruit_ID.Notes);

% GM-CSF effect parameters for APC recruitment and maturation
EC50_GMCSF = sbioselect(model,'Type','parameter','Name','EC50_GMCSF');
if isempty(EC50_GMCSF), EC50_GMCSF = addparameter(model,'EC50_GMCSF',params.EC50_GMCSF.Value,'ValueUnits',params.EC50_GMCSF.Units); end
set(EC50_GMCSF, 'Notes', params.EC50_GMCSF.Notes);

k_APC_mature_ID = sbioselect(model,'Type','parameter','Name','k_APC_mature_ID');
if isempty(k_APC_mature_ID), k_APC_mature_ID = addparameter(model,'k_APC_mature_ID',params.k_APC_mature_ID.Value,'ValueUnits',params.k_APC_mature_ID.Units); end
set(k_APC_mature_ID, 'Notes', params.k_APC_mature_ID.Notes);

% APC migration from injection site to lymph nodes
k_APC_mig_ID_LN = sbioselect(model,'Type','parameter','Name','k_APC_mig_ID_LN');
if isempty(k_APC_mig_ID_LN), k_APC_mig_ID_LN = addparameter(model,'k_APC_mig_ID_LN',params.k_APC_mig_ID_LN.Value,'ValueUnits',params.k_APC_mig_ID_LN.Units); end
set(k_APC_mig_ID_LN, 'Notes', params.k_APC_mig_ID_LN.Notes);

% APC death at injection site
k_APC_death_ID = sbioselect(model,'Type','parameter','Name','k_APC_death_ID');
if isempty(k_APC_death_ID), k_APC_death_ID = addparameter(model,'k_APC_death_ID',params.k_APC_death_ID.Value,'ValueUnits',params.k_APC_death_ID.Units); end
set(k_APC_death_ID, 'Notes', params.k_APC_death_ID.Notes);

% Antigen consumption per APC maturation
sigma_P1_per_mAPC = sbioselect(model,'Type','parameter','Name','sigma_P1_per_mAPC');
if isempty(sigma_P1_per_mAPC), sigma_P1_per_mAPC = addparameter(model,'sigma_P1_per_mAPC',params.sigma_P1_per_mAPC.Value,'ValueUnits',params.sigma_P1_per_mAPC.Units); end
set(sigma_P1_per_mAPC, 'Notes', params.sigma_P1_per_mAPC.Notes);

% EC50 for antigen-dependent APC maturation
EC50_P1_mature = sbioselect(model,'Type','parameter','Name','EC50_P1_mature');
if isempty(EC50_P1_mature), EC50_P1_mature = addparameter(model,'EC50_P1_mature',params.EC50_P1_mature.Value,'ValueUnits',params.EC50_P1_mature.Units); end
set(EC50_P1_mature, 'Notes', params.EC50_P1_mature.Notes);

% Capacity and baseline recruitment
K_APC_ID = sbioselect(model,'Type','parameter','Name','K_APC_ID');
if isempty(K_APC_ID), K_APC_ID = addparameter(model,'K_APC_ID',params.K_APC_ID.Value,'ValueUnits','cell'); end

k_APC_recruit0 = sbioselect(model,'Type','parameter','Name','k_APC_recruit0');
if isempty(k_APC_recruit0), k_APC_recruit0 = addparameter(model,'k_APC_recruit0',params.k_APC_recruit0.Value,'ValueUnits',params.k_APC_recruit0.Units); end

% Separate mature-DC death
k_mAPC_death_ID = sbioselect(model,'Type','parameter','Name','k_mAPC_death_ID');
if isempty(k_mAPC_death_ID), k_mAPC_death_ID = addparameter(model,'k_mAPC_death_ID',params.k_mAPC_death_ID.Value,'ValueUnits',params.k_mAPC_death_ID.Units); end

% Antigen yield per cell death and degradation/drain
y_Ag_per_cell = sbioselect(model,'Type','parameter','Name','y_Ag_per_cell');
if isempty(y_Ag_per_cell), y_Ag_per_cell = addparameter(model,'y_Ag_per_cell',params.y_Ag_per_cell.Value,'ValueUnits','nanomole/cell'); end

k_P1_deg_ID = sbioselect(model,'Type','parameter','Name','k_P1_deg_ID');
if isempty(k_P1_deg_ID), k_P1_deg_ID = addparameter(model,'k_P1_deg_ID',params.k_P1_deg_ID.Value,'ValueUnits',params.k_P1_deg_ID.Units); end

% APC migration saturation
K_APC_mig = sbioselect(model,'Type','parameter','Name','K_APC_mig');
if isempty(K_APC_mig), K_APC_mig = addparameter(model,'K_APC_mig',params.K_APC_mig.Value,'ValueUnits',params.K_APC_mig.Units); end

% Antigen-transfer rate: migratory mcDC2 -> resident cDC1 cross-licensing in LN
k_cDC1_Ag_transfer = sbioselect(model,'Type','parameter','Name','k_cDC1_Ag_transfer');
if isempty(k_cDC1_Ag_transfer), k_cDC1_Ag_transfer = addparameter(model,'k_cDC1_Ag_transfer',params.k_cDC1_Ag_transfer.Value,'ValueUnits',params.k_cDC1_Ag_transfer.Units); end
set(k_cDC1_Ag_transfer, 'Notes', params.k_cDC1_Ag_transfer.Notes);

%% ---- Add Hill function for GM-CSF effects at injection site ----
% Hill function for GM-CSF effects (normalized to [0,1])
H_GMCSF_ID = sbioselect(model,'Type','parameter','Name','H_GMCSF_ID');
if isempty(H_GMCSF_ID), H_GMCSF_ID = addparameter(model,'H_GMCSF_ID',0,'ValueUnits','dimensionless','ConstantValue',false); end

% Get lymph node compartment for APC migration
V_LN = sbioselect(model, 'Name', 'V_LN', 'Type', 'compartment');

%% ---- Add Hill function rule for injection site GM-CSF ----
gmcsfRuleTxt_ID = 'H_GMCSF_ID = (V_ID.GMCSF / V_ID) / (EC50_GMCSF + (V_ID.GMCSF / V_ID))';

% Replace existing rules if present, else add
R = sbioselect(model,'Type','rule');
didReplaceGMCSF_ID = false;
for k = 1:numel(R)
    if contains(strrep(R(k).Rule,' ',''), 'H_GMCSF_ID=')
        set(R(k),'Rule',gmcsfRuleTxt_ID); didReplaceGMCSF_ID = true;
    end
end
if ~didReplaceGMCSF_ID
    addrule(model, gmcsfRuleTxt_ID, 'repeatedAssignment');
end

%% ---- Add APC maturation flux parameter and rule ----
% APC maturation flux (used for both maturation and antigen consumption)
J_mature_ID = sbioselect(model,'Type','parameter','Name','J_mature_ID');
if isempty(J_mature_ID), J_mature_ID = addparameter(model,'J_mature_ID',0,'ValueUnits','cell/day','ConstantValue',false); end

% Define maturation flux rule
maturationFluxRuleTxt_ID = 'J_mature_ID = k_APC_mature_ID * H_GMCSF_ID * V_ID.APC * (V_ID.P1_GVAX / V_ID) / ((V_ID.P1_GVAX / V_ID) + EC50_P1_mature)';

% Replace existing rules if present, else add
didReplaceMaturation_ID = false;
for k = 1:numel(R)
    if contains(strrep(R(k).Rule,' ',''), 'J_mature_ID=')
        set(R(k),'Rule',maturationFluxRuleTxt_ID); didReplaceMaturation_ID = true;
    end
end
if ~didReplaceMaturation_ID
    addrule(model, maturationFluxRuleTxt_ID, 'repeatedAssignment');
end

%% ---- Injection site GVAX cell dynamics ----
% GVAX cell clearance at injection site
clearanceRxn = sbioselect(model, 'Type', 'reaction', 'Name', 'GVAX_clearance_ID');
if isempty(clearanceRxn)
    clearanceRxn = addreaction(model, 'V_ID.GVAX_cells -> null');
    set(clearanceRxn, 'Name', 'GVAX_clearance_ID');
    set(clearanceRxn, 'ReactionRate', 'k_GVAX_clearance * V_ID.GVAX_cells');
    set(clearanceRxn, 'Notes', 'GVAX cell clearance at injection site');
end

% GM-CSF production by GVAX cells at injection site
gmcsfProdRxn = sbioselect(model, 'Type', 'reaction', 'Name', 'GMCSF_production_ID');
if isempty(gmcsfProdRxn)
    gmcsfProdRxn = addreaction(model, 'null -> V_ID.GMCSF');
    set(gmcsfProdRxn, 'Name', 'GMCSF_production_ID');
    set(gmcsfProdRxn, 'ReactionRate', 'k_GMCSF_prod * V_ID.GVAX_cells');
    set(gmcsfProdRxn, 'Notes', 'GM-CSF production by GVAX cells at injection site');
end

% GM-CSF degradation at injection site
gmcsfDegRxn = sbioselect(model, 'Type', 'reaction', 'Name', 'GMCSF_degradation_ID');
if isempty(gmcsfDegRxn)
    gmcsfDegRxn = addreaction(model, 'V_ID.GMCSF -> null');
    set(gmcsfDegRxn, 'Name', 'GMCSF_degradation_ID');
    set(gmcsfDegRxn, 'ReactionRate', 'k_GMCSF_deg * V_ID.GMCSF');
    set(gmcsfDegRxn, 'Notes', 'GM-CSF degradation at injection site');
end

% Antigen release by GVAX cells at injection site
antigenReleaseRxn = sbioselect(model,'Type','reaction','Name','GVAX_antigen_release_ID');
if isempty(antigenReleaseRxn)
    antigenReleaseRxn = addreaction(model,'null -> V_ID.P1_GVAX');
    set(antigenReleaseRxn,'Name','GVAX_antigen_release_ID');
end
set(antigenReleaseRxn,'ReactionRate','y_Ag_per_cell * k_GVAX_clearance * V_ID.GVAX_cells');

% Antigen degradation/dispersion
p1DegRxn = sbioselect(model,'Type','reaction','Name','P1_degradation_ID');
if isempty(p1DegRxn)
    p1DegRxn = addreaction(model,'V_ID.P1_GVAX -> null');
    set(p1DegRxn,'Name','P1_degradation_ID');
end
set(p1DegRxn,'ReactionRate','k_P1_deg_ID * V_ID.P1_GVAX');

%% ---- APC dynamics at injection site ----
% APC recruitment to injection site (GM-CSF mediated)
apcRecruitRxn = sbioselect(model,'Type','reaction','Name','APC_recruitment_ID');
if isempty(apcRecruitRxn)
    apcRecruitRxn = addreaction(model,'null -> V_ID.APC');
    set(apcRecruitRxn,'Name','APC_recruitment_ID');
end
set(apcRecruitRxn,'ReactionRate',...
  '(k_APC_recruit0 + k_APC_recruit_ID * H_GMCSF_ID) * K_APC_ID /(K_APC_ID + V_ID.APC + V_ID.mAPC)');

% APC maturation at injection site (antigen + GM-CSF mediated)
apcMatureRxn = sbioselect(model, 'Type', 'reaction', 'Name', 'APC_maturation_ID');
if isempty(apcMatureRxn)
    apcMatureRxn = addreaction(model, 'V_ID.APC -> V_ID.mAPC');
    set(apcMatureRxn, 'Name', 'APC_maturation_ID');
    set(apcMatureRxn, 'ReactionRate', 'J_mature_ID');
    set(apcMatureRxn, 'Notes', 'APC maturation with antigen uptake at injection site');
end

% Antigen consumption tied to APC maturation flux
antigenConsumptionRxn = sbioselect(model, 'Type', 'reaction', 'Name', 'P1_consumption_ID');
if isempty(antigenConsumptionRxn)
    antigenConsumptionRxn = addreaction(model, 'V_ID.P1_GVAX -> null');
    set(antigenConsumptionRxn, 'Name', 'P1_consumption_ID');
    set(antigenConsumptionRxn, 'ReactionRate', 'sigma_P1_per_mAPC * J_mature_ID');
    set(antigenConsumptionRxn, 'Notes', 'Antigen consumption tied to APC maturation events');
end

% APC death at injection site
apcDeathRxn = sbioselect(model, 'Type', 'reaction', 'Name', 'APC_death_ID');
if isempty(apcDeathRxn)
    apcDeathRxn = addreaction(model, 'V_ID.APC -> null');
    set(apcDeathRxn, 'Name', 'APC_death_ID');
    set(apcDeathRxn, 'ReactionRate', 'k_APC_death_ID * V_ID.APC');
    set(apcDeathRxn, 'Notes', 'APC death at injection site');
end

% Mature APC death at injection site
mapcDeathRxn = sbioselect(model,'Type','reaction','Name','mAPC_death_ID');
if isempty(mapcDeathRxn)
    mapcDeathRxn = addreaction(model,'V_ID.mAPC -> null');
    set(mapcDeathRxn,'Name','mAPC_death_ID');
end
set(mapcDeathRxn,'ReactionRate','k_mAPC_death_ID * V_ID.mAPC');

%% ---- APC migration from injection site to lymph nodes ----
if ~isempty(V_LN)
  apcMigrationRxn = sbioselect(model,'Type','reaction','Name','mAPC_migration_ID_LN');
  if isempty(apcMigrationRxn)
      apcMigrationRxn = addreaction(model,'V_ID.mAPC -> V_LN.mcDC2');
      set(apcMigrationRxn,'Name','mAPC_migration_ID_LN');
  end
  set(apcMigrationRxn,'ReactionRate',...
      'k_APC_mig_ID_LN * V_ID.mAPC / (1 + V_LN.mcDC2 / K_APC_mig)');

  %% ---- Antigen transfer: migratory mcDC2 -> LN-resident cDC1 cross-licensing ----
  % GM-CSF-derived DC vaccines (GVAX) prime CD8 predominantly via antigen
  % handoff to LN-resident Batf3+ cDC1, not by direct cross-presentation
  % (Ferris 2022 PMID: 35648641). GMDC vaccines fail in Irf8+32-/- mice
  % lacking host cDC1. This reaction captures that route: antigen-loaded
  % mcDC2 arriving in LN drives resident cDC1 -> mcDC1 maturation.
  cDC1TransferRxn = sbioselect(model,'Type','reaction','Name','cDC1_Ag_transfer_LN');
  if isempty(cDC1TransferRxn)
      cDC1TransferRxn = addreaction(model,'V_LN.cDC1 -> V_LN.mcDC1');
      set(cDC1TransferRxn,'Name','cDC1_Ag_transfer_LN');
      set(cDC1TransferRxn,'Notes','Antigen transfer from GVAX-derived mcDC2 to LN-resident cDC1 (Ferris 2022)');
  end
  set(cDC1TransferRxn,'ReactionRate',...
      'k_cDC1_Ag_transfer * V_LN.cDC1 * V_LN.mcDC2');
end

%% ---- Optional: Cyclophosphamide for Treg depletion ----
if add_cy
    fprintf('   Adding cyclophosphamide for Treg depletion...\n');

    % Get compartments
    V_C = sbioselect(model, 'Name', 'V_C', 'Type', 'compartment');
    V_T = sbioselect(model, 'Name', 'V_T', 'Type', 'compartment');

    if ~isempty(V_C)
        % Add Cy species in central compartment (target for dosing)
        cy_species = sbioselect(model, 'Name', 'Cy', 'Type', 'species', 'Parent', V_C);
        if isempty(cy_species)
            cy_species = addspecies(V_C, 'Cy', 0, 'InitialAmountUnits', 'milligram');
            set(cy_species, 'Notes', 'Cyclophosphamide for Treg depletion (low-dose immunomodulatory)');
        end

        % Add parameters
        k_Cy_clear = addparameter(model, 'k_Cy_clear', params.k_Cy_clear.Value, ...
            'ValueUnits', params.k_Cy_clear.Units, 'Notes', params.k_Cy_clear.Notes);
        k_Cy_Treg_deplete = addparameter(model, 'k_Cy_Treg_deplete', params.k_Cy_Treg_deplete.Value, ...
            'ValueUnits', params.k_Cy_Treg_deplete.Units, 'Notes', params.k_Cy_Treg_deplete.Notes);
        Cy_50_Treg = addparameter(model, 'Cy_50_Treg', params.Cy_50_Treg.Value, ...
            'ValueUnits', params.Cy_50_Treg.Units, 'Notes', params.Cy_50_Treg.Notes);

        % Cy clearance/degradation
        cyDegRxn = sbioselect(model, 'Type', 'reaction', 'Name', 'Cy_degradation');
        if isempty(cyDegRxn)
            cyDegRxn = addreaction(model, 'V_C.Cy -> null');
            set(cyDegRxn, 'Name', 'Cy_degradation');
            set(cyDegRxn, 'Notes', 'Cyclophosphamide clearance (t1/2 ~12-24 hours)');
        end
        set(cyDegRxn, 'ReactionRate', 'k_Cy_clear * V_C.Cy');

        % Cy-mediated Treg depletion in tumor
        if ~isempty(V_T)
            treg_species = sbioselect(model, 'Name', 'Treg', 'Type', 'species', 'Parent', V_T);
            if ~isempty(treg_species)
                cyTregDepRxn = sbioselect(model, 'Type', 'reaction', 'Name', 'Cy_Treg_depletion');
                if isempty(cyTregDepRxn)
                    cyTregDepRxn = addreaction(model, 'V_T.Treg -> null');
                    set(cyTregDepRxn, 'Name', 'Cy_Treg_depletion');
                    set(cyTregDepRxn, 'Notes', 'Selective Treg depletion by low-dose cyclophosphamide');
                end
                % Hill function: depletion increases with Cy concentration
                set(cyTregDepRxn, 'ReactionRate', ...
                    'k_Cy_Treg_deplete * V_T.Treg * (V_C.Cy / (Cy_50_Treg + V_C.Cy))');
            end
        end

        fprintf('   Cyclophosphamide module added successfully\n');
    else
        warning('Could not add cyclophosphamide: V_C compartment not found');
    end
end

end


