% Dose Schedule
%
% Generates Donsing Schedule
%
% Inputs: drugName -- drug name or character array of drug names
%         varargin -- Name-Value Pairs
%                     - drugName_dose (mg/kg)
%                     - drugName_schedule [start,interval,repeat]
%                     - patientWeight (kg)
%
%       vaild drugName values: nivolumab, atezolizumab, ipilimumab, pembrolizumab, entinostat, nab-paclitaxel, tremelimumab, GVAX, Cy
%
% Outputs: dosing -- SimBiology model object with new antigen module


function dose_schedule = schedule_dosing(drugName,varargin)

% Check if drugName is cell array
if (iscell(drugName))
    N = length(drugName);
else
    N = 1;
    drugName = {drugName};
end

% Optional Inputs
in = inputParser;
% Pembrolizumab
addParameter(in,'pembrolizumab_dose',2.5); % 200 mg every three weeks
addParameter(in,'pembrolizumab_schedule',[0,21,30]);
% Nivolumab
addParameter(in,'nivolumab_dose',20); % 20 mg/kg every four weeks
addParameter(in,'nivolumab_schedule',[0,28,30]);
% Atezolizumab
addParameter(in,'atezolizumab_dose',15); % 15 mg/kg every three weeks
addParameter(in,'atezolizumab_schedule',[0,21,30]);
% Durvalumab
addParameter(in,'durvalumab_dose',10); % 10 g/kg every two weeks
addParameter(in,'durvalumab_schedule',[0,14,30]);
% Ipilimumab
addParameter(in,'ipilimumab_dose',1); % 1 mg/kg every three weeks
addParameter(in,'ipilimumab_schedule',[0,21,30]);
% Tremelimumab
addParameter(in,'tremelimumab_dose',1); % 1 mg/kg every four weeks
addParameter(in,'tremelimumab_schedule',[0,28,30]);
% Entinostat
addParameter(in,'entinostat_dose',3); % 3 mg every week
addParameter(in,'entinostat_schedule',[0,7,55]);
% NabPaclitaxel
addParameter(in,'nabPaclitaxel_dose',100); % 100 mg/m2 Q3/4W
addParameter(in,'nabPaclitaxel_schedule',[0,28,15]);
% ---- PASS-01 cytotoxic chemotherapy (mg/m2; schedule [start,interval,repeat] in days) ----
% Gemcitabine (GnP): 1000 mg/m2, 30-min IV, days 1/8/15 q28d
addParameter(in,'gemcitabine_dose',1000);
addParameter(in,'gemcitabine_schedule',[0,28,6]);
% 5-Fluorouracil (mFOLFIRINOX): 2400 mg/m2, 46-h continuous IV, q14d (bolus dropped)
addParameter(in,'fluorouracil_dose',2400);
addParameter(in,'fluorouracil_schedule',[0,14,12]);
% Oxaliplatin (mFOLFIRINOX): 85 mg/m2, 2-h IV, q14d
addParameter(in,'oxaliplatin_dose',85);
addParameter(in,'oxaliplatin_schedule',[0,14,12]);
% Irinotecan (mFOLFIRINOX): 150 mg/m2, 90-min IV, q14d
addParameter(in,'irinotecan_dose',150);
addParameter(in,'irinotecan_schedule',[0,14,12]);
% Cabozantinib - 40mg for combined therapy; 60mg for mono
addParameter(in,'cabozantinib_dose',40); % 40 mg daily % [0,1,1] one dose [0,1,10] 10 doses
% https://www.accessdata.fda.gov/drugsatfda_docs/label/2012/203756lbl.pdf
addParameter(in,'cabozantinib_schedule',[0,1,400]); % 70 days Dosing
% aCD47
addParameter(in,'aCD47_dose',10); % 10 mg/kg in combo; 15 in mono
addParameter(in,'aCD47_schedule',[0,7,57]);
% GVAX
addParameter(in,'GVAX_dose',5e8); % 5e8 cells per vaccination (typical clinical dose)
addParameter(in,'GVAX_schedule',[14,21,6]); % Prime at day 14, boost every 21 days, 6 total doses
% Cyclophosphamide (low-dose for Treg depletion). Both names are accepted
% so scenario YAMLs can use the unambiguous 'cyclophosphamide_*' form
% (matching the drug name in drug_metadata.yaml / the C++ side).
addParameter(in,'Cy_dose',380); % mg (final dose, already BSA-multiplied — e.g. 200 mg/m² × 1.9 m² BSA)
addParameter(in,'Cy_schedule',[13,21,6]); % Day 13 (1 day before GVAX), every 21 days, 6 doses
addParameter(in,'cyclophosphamide_dose',380);  % alias for Cy_dose
addParameter(in,'cyclophosphamide_schedule',[13,21,6]);  % alias for Cy_schedule
% Patient Weight
addParameter(in,'patientWeight',80);
% Patient Body Surface area
addParameter(in,'patientBSA',1.9);
% Parse Inputs
parse(in,varargin{:});
% Pembrolizumab
dose_pembro = in.Results.pembrolizumab_dose;
schedule_pembro = in.Results.pembrolizumab_schedule;
% Nivolumab
dose_nivo = in.Results.nivolumab_dose;
schedule_nivo = in.Results.nivolumab_schedule;
% Atezolizumab
dose_atezo = in.Results.atezolizumab_dose;
schedule_atezo = in.Results.atezolizumab_schedule;
% Durvalumab
dose_durva = in.Results.durvalumab_dose;
schedule_durva = in.Results.durvalumab_schedule;
% Ipilimumab
dose_ipil = in.Results.ipilimumab_dose;
schedule_ipil = in.Results.ipilimumab_schedule;
% Tremelimumab
dose_treme = in.Results.tremelimumab_dose;
schedule_treme = in.Results.tremelimumab_schedule;
% Entinostat
dose_ENT = in.Results.entinostat_dose;
schedule_ENT = in.Results.entinostat_schedule;
% NabPaclitaxel
dose_nabp = in.Results.nabPaclitaxel_dose;
schedule_nabp = in.Results.nabPaclitaxel_schedule;
% Cytotoxic chemotherapy
dose_gem = in.Results.gemcitabine_dose;   schedule_gem = in.Results.gemcitabine_schedule;
dose_fu5 = in.Results.fluorouracil_dose;  schedule_fu5 = in.Results.fluorouracil_schedule;
dose_oxa = in.Results.oxaliplatin_dose;   schedule_oxa = in.Results.oxaliplatin_schedule;
dose_iri = in.Results.irinotecan_dose;    schedule_iri = in.Results.irinotecan_schedule;
% Cabozantinib
dose_cabo = in.Results.cabozantinib_dose;
schedule_cabo = in.Results.cabozantinib_schedule;
% aCD47
dose_aCD47 = in.Results.aCD47_dose;
schedule_aCD47 = in.Results.aCD47_schedule;
% GVAX
dose_GVAX = in.Results.GVAX_dose;
schedule_GVAX = in.Results.GVAX_schedule;
% Cyclophosphamide — prefer the explicit cyclophosphamide_* name if the
% caller provided it (scenario YAMLs use this form), otherwise fall back
% to the legacy Cy_* name. `UsingDefaults` lists any parameter left at
% its default so this check distinguishes "caller set cyclophosphamide"
% from "caller set Cy".
if any(strcmp(in.UsingDefaults, 'cyclophosphamide_dose'))
    dose_Cy = in.Results.Cy_dose;
else
    dose_Cy = in.Results.cyclophosphamide_dose;
end
if any(strcmp(in.UsingDefaults, 'cyclophosphamide_schedule'))
    schedule_Cy = in.Results.Cy_schedule;
else
    schedule_Cy = in.Results.cyclophosphamide_schedule;
end
% Patient Weight
patient_weight = in.Results.patientWeight;
% Patient BSA
patient_BSA = in.Results.patientBSA;

% Pembrolizumab
MW_pembro = 1.49E8;
doseObj_pembro = sbiodose('pembro','Amount',patient_weight*dose_pembro/MW_pembro,'AmountUnits','mole','TargetName','V_C.aPD1');
doseObj_pembro.StartTime = schedule_pembro(1);
doseObj_pembro.Interval = schedule_pembro(2);
doseObj_pembro.TimeUnits = 'day';
doseObj_pembro.RepeatCount = schedule_pembro(3);
doseObj_pembro.Active = true;

% Nivolumab
MW_nivo = 1.436E8; % milligrams per mole
doseObj_nivo = sbiodose('nivo','Amount',patient_weight*dose_nivo/MW_nivo,'AmountUnits','mole','TargetName','V_C.aPD1');
doseObj_nivo.StartTime = schedule_nivo(1);
doseObj_nivo.Interval = schedule_nivo(2);
doseObj_nivo.TimeUnits = 'day';
doseObj_nivo.RepeatCount = schedule_nivo(3);
doseObj_nivo.Active = true;

% Entinostat
MW_ENT = 3.764085e5; % mg/mole
Bio = 0.18; % Fraction of dose through buccal absorption

doseObj_ENT_1 = sbiodose('ENT_1','Amount',Bio*dose_ENT/MW_ENT,'AmountUnits','mole','TargetName','V_C.inostat_Buccal');
doseObj_ENT_1.StartTime = schedule_ENT(1);
doseObj_ENT_1.Interval = schedule_ENT(2);
doseObj_ENT_1.TimeUnits = 'day';
doseObj_ENT_1.RepeatCount = schedule_ENT(3);
doseObj_ENT_1.Active = true;
doseObj_ENT_1.DurationParameterName = 'durP_inostat';

doseObj_ENT_2 = sbiodose('ENT_2','Amount',(1-Bio)*dose_ENT/MW_ENT,'AmountUnits','mole','TargetName','V_C.Dose_GI');
doseObj_ENT_2.StartTime = schedule_ENT(1);
doseObj_ENT_2.Interval = schedule_ENT(2);
doseObj_ENT_2.TimeUnits = 'day';
doseObj_ENT_2.RepeatCount = schedule_ENT(3);
doseObj_ENT_2.Active = true;
doseObj_ENT_2.LagParameterName = 'lagP_inostat';

% nabPaclitaxel
% MW = 853.9; % g/mol
doseObj_nabp_1 = sbiodose('nabp_1','Amount',dose_nabp*patient_BSA,'AmountUnits','milligram','TargetName','V_1.NabP');
doseObj_nabp_1.Rate = doseObj_nabp_1.Amount/30;
doseObj_nabp_1.RateUnits = 'milligram/minute';
doseObj_nabp_1.StartTime = schedule_nabp(1);
doseObj_nabp_1.Active = true;
doseObj_nabp_1.Interval = schedule_nabp(2);
doseObj_nabp_1.TimeUnits = 'day';
doseObj_nabp_1.RepeatCount = schedule_nabp(3);

doseObj_nabp_2 = sbiodose('nabp_2','Amount',dose_nabp*patient_BSA,'AmountUnits','milligram','TargetName','V_1.NabP');
doseObj_nabp_2.Rate = doseObj_nabp_2.Amount/30;
doseObj_nabp_2.RateUnits = 'milligram/minute';
doseObj_nabp_2.StartTime = schedule_nabp(1)+7;
doseObj_nabp_2.Active = true;
doseObj_nabp_2.Interval = schedule_nabp(2);
doseObj_nabp_2.TimeUnits = 'day';
doseObj_nabp_2.RepeatCount = schedule_nabp(3);

doseObj_nabp_3 = sbiodose('nabp_3','Amount',dose_nabp*patient_BSA,'AmountUnits','milligram','TargetName','V_1.NabP');
doseObj_nabp_3.Rate = doseObj_nabp_3.Amount/30;
doseObj_nabp_3.RateUnits = 'milligram/minute';
doseObj_nabp_3.StartTime = schedule_nabp(1)+14;
doseObj_nabp_3.Active = true;
doseObj_nabp_3.Interval = schedule_nabp(2);
doseObj_nabp_3.TimeUnits = 'day';
doseObj_nabp_3.RepeatCount = schedule_nabp(3);

% ---- Cytotoxic chemotherapy (mg dose = mg/m2 x BSA; zero-order IV infusion) ----
% Gemcitabine: days 1/8/15 q28d, 30-min infusion into C_gem.gem (3 dose objects,
% each repeating every 28 d, offset 0/7/14 -> the d1/8/15 pattern; like nab-paclitaxel)
doseObj_gem_1 = sbiodose('gem_1','Amount',dose_gem*patient_BSA,'AmountUnits','milligram','TargetName','C_gem.gem');
doseObj_gem_1.Rate = doseObj_gem_1.Amount/30; doseObj_gem_1.RateUnits = 'milligram/minute';
doseObj_gem_1.StartTime = schedule_gem(1); doseObj_gem_1.Interval = schedule_gem(2); doseObj_gem_1.TimeUnits = 'day'; doseObj_gem_1.RepeatCount = schedule_gem(3); doseObj_gem_1.Active = true;
doseObj_gem_2 = sbiodose('gem_2','Amount',dose_gem*patient_BSA,'AmountUnits','milligram','TargetName','C_gem.gem');
doseObj_gem_2.Rate = doseObj_gem_2.Amount/30; doseObj_gem_2.RateUnits = 'milligram/minute';
doseObj_gem_2.StartTime = schedule_gem(1)+7; doseObj_gem_2.Interval = schedule_gem(2); doseObj_gem_2.TimeUnits = 'day'; doseObj_gem_2.RepeatCount = schedule_gem(3); doseObj_gem_2.Active = true;
doseObj_gem_3 = sbiodose('gem_3','Amount',dose_gem*patient_BSA,'AmountUnits','milligram','TargetName','C_gem.gem');
doseObj_gem_3.Rate = doseObj_gem_3.Amount/30; doseObj_gem_3.RateUnits = 'milligram/minute';
doseObj_gem_3.StartTime = schedule_gem(1)+14; doseObj_gem_3.Interval = schedule_gem(2); doseObj_gem_3.TimeUnits = 'day'; doseObj_gem_3.RepeatCount = schedule_gem(3); doseObj_gem_3.Active = true;

% 5-Fluorouracil: 46-h continuous infusion q14d into C_fu5.fu5
doseObj_fu5 = sbiodose('fu5','Amount',dose_fu5*patient_BSA,'AmountUnits','milligram','TargetName','C_fu5.fu5');
doseObj_fu5.Rate = doseObj_fu5.Amount/(46*60); doseObj_fu5.RateUnits = 'milligram/minute';
doseObj_fu5.StartTime = schedule_fu5(1); doseObj_fu5.Interval = schedule_fu5(2); doseObj_fu5.TimeUnits = 'day'; doseObj_fu5.RepeatCount = schedule_fu5(3); doseObj_fu5.Active = true;

% Oxaliplatin: 2-h infusion q14d into C_oxa.oxa
doseObj_oxa = sbiodose('oxa','Amount',dose_oxa*patient_BSA,'AmountUnits','milligram','TargetName','C_oxa.oxa');
doseObj_oxa.Rate = doseObj_oxa.Amount/120; doseObj_oxa.RateUnits = 'milligram/minute';
doseObj_oxa.StartTime = schedule_oxa(1); doseObj_oxa.Interval = schedule_oxa(2); doseObj_oxa.TimeUnits = 'day'; doseObj_oxa.RepeatCount = schedule_oxa(3); doseObj_oxa.Active = true;

% Irinotecan: 90-min infusion q14d into C_iri.iri (parent; SN-38 formed in-model)
doseObj_iri = sbiodose('iri','Amount',dose_iri*patient_BSA,'AmountUnits','milligram','TargetName','C_iri.iri');
doseObj_iri.Rate = doseObj_iri.Amount/90; doseObj_iri.RateUnits = 'milligram/minute';
doseObj_iri.StartTime = schedule_iri(1); doseObj_iri.Interval = schedule_iri(2); doseObj_iri.TimeUnits = 'day'; doseObj_iri.RepeatCount = schedule_iri(3); doseObj_iri.Active = true;

% Cabozantinib
MW_cabo = 6.356E5; % milligrams per mole % https://www.accessdata.fda.gov/drugsatfda_docs/label/2012/203756lbl.pdf
F_cabo = 0.854; % fraction of dose in the absorption site

doseObj_cabo_1 = sbiodose('cabo_1','Amount',F_cabo*dose_cabo/MW_cabo,'AmountUnits','mole','TargetName','V_C.A_site1');
doseObj_cabo_1.StartTime = schedule_cabo(1);
doseObj_cabo_1.Interval = schedule_cabo(2);
doseObj_cabo_1.TimeUnits = 'day';
doseObj_cabo_1.RepeatCount = schedule_cabo(3);
doseObj_cabo_1.Active = true;
doseObj_cabo_1.LagParameterName = 'lagP1_cabozantinib';

% doseObj_cabo_2 = sbiodose('cabo_2','Amount',(1-F_cabo)*dose_cabo/MW_cabo,'AmountUnits','mole','TargetName','V_C.cabozantinib_absorption_site');
doseObj_cabo_2 = sbiodose('cabo_2','Amount',(1-F_cabo)*dose_cabo/MW_cabo,'AmountUnits','mole','TargetName','V_C.A_site2');
doseObj_cabo_2.StartTime = schedule_cabo(1);
doseObj_cabo_2.Interval = schedule_cabo(2);
doseObj_cabo_2.TimeUnits = 'day';
doseObj_cabo_2.RepeatCount = schedule_cabo(3);
doseObj_cabo_2.Active = true;
doseObj_cabo_2.LagParameterName = 'lagP2_cabozantinib';
% doseObj_cabo_2.DurationParameterName = 'durP_cabozantinib';


% Atezolizumab
MW_atezo = 1.436E8; % milligrams per mole
doseObj_atezo = sbiodose('atezo','Amount',patient_weight*dose_atezo/MW_atezo,'AmountUnits','mole','TargetName','V_C.aPDL1');
doseObj_atezo.StartTime = schedule_atezo(1);
doseObj_atezo.Interval = schedule_atezo(2);
doseObj_atezo.TimeUnits = 'day';
doseObj_atezo.RepeatCount = schedule_atezo(3);
doseObj_atezo.Active = true;

% Durvalumab
MW_durva = 1.49E8; % milligrams per mole
doseObj_durva = sbiodose('durva','Amount',patient_weight*dose_durva/MW_durva,'AmountUnits','mole','TargetName','V_C.aPDL1');
doseObj_durva.StartTime = schedule_durva(1);
doseObj_durva.Interval = schedule_durva(2);
doseObj_durva.TimeUnits = 'day';
doseObj_durva.RepeatCount = schedule_durva(3);
doseObj_durva.Active = true;

% aCD47
MW_aCD47 = 7.8E7; % milligrams per mole
doseObj_aCD47 = sbiodose('aCD47','Amount',patient_weight*dose_aCD47/MW_aCD47,'AmountUnits','mole','TargetName','V_C.aCD47');
doseObj_aCD47.StartTime = schedule_aCD47(1);
doseObj_aCD47.Rate = doseObj_aCD47.Amount/60;
doseObj_aCD47.RateUnits = 'mole/minute';
doseObj_aCD47.Interval = schedule_aCD47(2);
doseObj_aCD47.TimeUnits = 'day';
doseObj_aCD47.RepeatCount = schedule_aCD47(3);
doseObj_aCD47.Active = true;

% Ipilimumab
MW_ipil = 1.486349E8; % milligrams per mole
doseObj_ipi = sbiodose('ipi','Amount',patient_weight*dose_ipil/MW_ipil,'AmountUnits','mole','TargetName','V_C.aCTLA4');
doseObj_ipi.StartTime = schedule_ipil(1);
doseObj_ipi.Interval = schedule_ipil(2);
doseObj_ipi.TimeUnits = 'day';
doseObj_ipi.RepeatCount = schedule_ipil(3);
doseObj_ipi.Active = true;

% Tremelimumab
MW_treme = 1.4638E8; % milligrams per mole
doseObj_treme = sbiodose('treme','Amount',patient_weight*dose_treme/MW_treme,'AmountUnits','mole','TargetName','V_C.aCTLA4');
doseObj_treme.StartTime = schedule_treme(1);
doseObj_treme.Interval = schedule_treme(2);
doseObj_treme.TimeUnits = 'day';
doseObj_treme.RepeatCount = schedule_treme(3);
doseObj_treme.Active = true;

% GVAX
doseObj_GVAX = sbiodose('GVAX','Amount',dose_GVAX,'AmountUnits','cell','TargetName','V_ID.GVAX_cells');
doseObj_GVAX.StartTime = schedule_GVAX(1);
doseObj_GVAX.Interval = schedule_GVAX(2);
doseObj_GVAX.TimeUnits = 'day';
doseObj_GVAX.RepeatCount = schedule_GVAX(3);
doseObj_GVAX.Active = true;

% Cyclophosphamide (low-dose for Treg depletion). dose_Cy is the final
% dose in mg — scenario YAMLs already pre-multiply by BSA (see
% drug_metadata.yaml, dose_basis: direct). Do not multiply by patient_BSA
% again here or the infusion is ~2× too large.
doseObj_Cy = sbiodose('Cy','Amount',dose_Cy,'AmountUnits','milligram','TargetName','V_C.Cy');
doseObj_Cy.StartTime = schedule_Cy(1);
doseObj_Cy.Interval = schedule_Cy(2);
doseObj_Cy.TimeUnits = 'day';
doseObj_Cy.RepeatCount = schedule_Cy(3);
doseObj_Cy.Active = true;

% Dose Schedule Array
dose_schedule(N) = sbiodose('empty'); % preallocate array
for i = 1:N
    switch drugName{i}
        case 'nivolumab'
            dose_schedule(i) = doseObj_nivo;
        case 'pembrolizumab'
            dose_schedule(i) = doseObj_pembro;
        case 'atezolizumab'
            dose_schedule(i) = doseObj_atezo;
        case 'durvalumab'
            dose_schedule(i) = doseObj_durva;
        case 'ipilimumab'
            dose_schedule(i) = doseObj_ipi;
        case 'tremelimumab'
            dose_schedule(i) = doseObj_treme;
        case 'nabPaclitaxel'
            dose_schedule(i) = doseObj_nabp_1;
        case 'gemcitabine'
            dose_schedule(i) = doseObj_gem_1;
        case 'fluorouracil'
            dose_schedule(i) = doseObj_fu5;
        case 'oxaliplatin'
            dose_schedule(i) = doseObj_oxa;
        case 'irinotecan'
            dose_schedule(i) = doseObj_iri;
        case 'cabozantinib'
            dose_schedule(i) = doseObj_cabo_1;
        case 'entinostat'
            dose_schedule(i) = doseObj_ENT_1;
        case 'aCD47'
            dose_schedule(i) = doseObj_aCD47;
        case 'GVAX'
            dose_schedule(i) = doseObj_GVAX;
        case {'Cy','cyclophosphamide','Cyclophosphamide'}
            dose_schedule(i) = doseObj_Cy;
        otherwise
            error('No match for drug name');
    end
end

for i = 1:N
    switch drugName{i}
        case 'nabPaclitaxel'
            dose_schedule(end+1) = doseObj_nabp_2;
            dose_schedule(end+1) = doseObj_nabp_3;
        case 'gemcitabine'
            dose_schedule(end+1) = doseObj_gem_2;
            dose_schedule(end+1) = doseObj_gem_3;
        case 'entinostat'
            dose_schedule(end+1) = doseObj_ENT_2;
        case 'cabozantinib'
            dose_schedule(end+1) = doseObj_cabo_2;
    end
end
