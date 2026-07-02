% Default dose regimen for atezolizumab is 1200 mg Q3W
% Default dose regimen for nab-paclitaxel is 100 mg/m2 Q3/4W

% dose_schedule = [];
% dose_schedule = schedule_dosing({'atezolizumab'});
% dose_schedule = schedule_dosing({'nabPaclitaxel'});
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[0,14,30], 'nabPaclitaxel_schedule',[0,28,15]);
% dose_schedule = schedule_dosing({'nivolumab','cabozantinib'}, 'nivolumab_dose', 3, 'nivolumab_schedule',[0,14,30]);
dose_schedule = schedule_dosing({'nivolumab'}, 'nivolumab_dose', 3, 'nivolumab_schedule',[0,14,30]);
% dose_schedule = schedule_dosing({'cabozantinib'});
% Sequential Therapy
% dose_schedule = schedule_dosing({'atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[0,14,30]);
% dose_schedule = schedule_dosing({'atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[14,14,30]);
% dose_schedule = schedule_dosing({'atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[28,14,30]);

% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[14,14,30], 'nabPaclitaxel_schedule',[0,28,15]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[28,14,30], 'nabPaclitaxel_schedule',[0,28,15]);
% dose_schedule = schedule_dosing({'nabPaclitaxel'}, 'nabPaclitaxel_schedule',[14,28,15]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[0,14,30], 'nabPaclitaxel_schedule',[14,28,15]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[14,14,30], 'nabPaclitaxel_schedule',[14,28,15]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[28,14,30], 'nabPaclitaxel_schedule',[14,28,15]);
% dose_schedule = schedule_dosing({'nabPaclitaxel'}, 'nabPaclitaxel_schedule',[28,28,15]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[0,14,30], 'nabPaclitaxel_schedule',[28,28,15]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[14,14,30], 'nabPaclitaxel_schedule',[28,28,15]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[28,14,30], 'nabPaclitaxel_schedule',[28,28,15]);

% dose_schedule = schedule_dosing({'nabPaclitaxel'}, 'nabPaclitaxel_dose',125, 'nabPaclitaxel_schedule',[0,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[0,14,30], 'nabPaclitaxel_dose',125, 'nabPaclitaxel_schedule',[0,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[14,14,30], 'nabPaclitaxel_dose',125, 'nabPaclitaxel_schedule',[0,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[28,14,30], 'nabPaclitaxel_dose',125, 'nabPaclitaxel_schedule',[0,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel'}, 'nabPaclitaxel_dose',125, 'nabPaclitaxel_schedule',[14,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[0,14,30], 'nabPaclitaxel_dose',125, 'nabPaclitaxel_schedule',[14,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[14,14,30], 'nabPaclitaxel_dose',125, 'nabPaclitaxel_schedule',[14,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[28,14,30], 'nabPaclitaxel_dose',125, 'nabPaclitaxel_schedule',[14,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel'}, 'nabPaclitaxel_dose',125, 'nabPaclitaxel_schedule',[28,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[0,14,30], 'nabPaclitaxel_dose',125, 'nabPaclitaxel_schedule',[28,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[14,14,30], 'nabPaclitaxel_dose',125, 'nabPaclitaxel_schedule',[28,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[28,14,30], 'nabPaclitaxel_dose',125, 'nabPaclitaxel_schedule',[28,21,20]);

% dose_schedule = schedule_dosing({'nabPaclitaxel'}, 'nabPaclitaxel_dose',260, 'nabPaclitaxel_schedule',[0,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[0,14,30], 'nabPaclitaxel_dose',260, 'nabPaclitaxel_schedule',[0,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[14,14,30], 'nabPaclitaxel_dose',260, 'nabPaclitaxel_schedule',[0,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[28,14,30], 'nabPaclitaxel_dose',260, 'nabPaclitaxel_schedule',[0,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel'}, 'nabPaclitaxel_dose',260, 'nabPaclitaxel_schedule',[14,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[0,14,30], 'nabPaclitaxel_dose',260, 'nabPaclitaxel_schedule',[14,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[14,14,30], 'nabPaclitaxel_dose',260, 'nabPaclitaxel_schedule',[14,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[28,14,30], 'nabPaclitaxel_dose',260, 'nabPaclitaxel_schedule',[14,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel'}, 'nabPaclitaxel_dose',260, 'nabPaclitaxel_schedule',[28,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[0,14,30], 'nabPaclitaxel_dose',260, 'nabPaclitaxel_schedule',[28,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[14,14,30], 'nabPaclitaxel_dose',260, 'nabPaclitaxel_schedule',[28,21,20]);
% dose_schedule = schedule_dosing({'nabPaclitaxel','atezolizumab'}, 'atezolizumab_dose',840/80, 'atezolizumab_schedule',[28,14,30], 'nabPaclitaxel_dose',260, 'nabPaclitaxel_schedule',[28,21,20]);
