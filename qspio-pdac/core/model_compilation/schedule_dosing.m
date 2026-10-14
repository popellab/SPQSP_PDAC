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
%       valid drugName values: nivolumab, pembrolizumab, GVAX, Cy
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

% GVAX
doseObj_GVAX = sbiodose('GVAX','Amount',dose_GVAX,'AmountUnits','cell','TargetName','V_ID.GVAX_cells');
doseObj_GVAX.StartTime = schedule_GVAX(1);
doseObj_GVAX.Interval = schedule_GVAX(2);
doseObj_GVAX.TimeUnits = 'day';
doseObj_GVAX.RepeatCount = schedule_GVAX(3);
doseObj_GVAX.Active = true;

% Cyclophosphamide (low-dose for Treg depletion). dose_Cy is the final
% dose in mg — scenario YAMLs already pre-multiply by BSA (see
% drug_metadata.yaml, dose_basis: direct). Do not multiply this by BSA again
% here or the infusion is ~2× too large.
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
        case 'GVAX'
            dose_schedule(i) = doseObj_GVAX;
        case {'Cy','cyclophosphamide','Cyclophosphamide'}
            dose_schedule(i) = doseObj_Cy;
        otherwise
            error('No match for drug name');
    end
end
