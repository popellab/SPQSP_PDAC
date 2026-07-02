% Metabolic Competition Module (Glucose/Lactate)
%
% Models tumor glucose consumption (Warburg effect) and lactate accumulation.
% Glucose depletion impairs T cell effector function; lactate promotes M2
% macrophage polarization. Creates positive feedback:
%   C1 growth -> glucose depletion -> T cell impairment -> reduced killing -> more C1
% Connects vasculature (K) to immune function through nutrient delivery.
%
% Requires: cancer_module (K, VEGF, C_total), Teff_module (T_total, k_C_Tcell_eff rule),
%           Th_module (IFNg secretion), macrophage_module (M_total, M1->M2 reaction),
%           MDSC_module, NK_module, hypoxia_module (HIF in M1->M2 rate, VEGF_norm_50)
% Forward-references: aPSC_total (fibroblast_module_enhanced)
%
% Inputs: model  -- SimBiology model object
%         params -- object containing model parameter Values, Units, and Notes
%
% Outputs: model -- SimBiology model object with metabolic competition

function model = metabolism_module(model, params)

%% Add Parameters
p = addparameter(model,'glc_normal',params.glc_normal.Value,'ValueUnits',params.glc_normal.Units, 'Notes', 'Normal pancreas tissue glucose concentration (micromolarity, Battini 2017)');
p.UserData.Source = params.glc_normal.Notes;
    p.UserData.derivedFrom = params.glc_normal.derivedFrom;

p = addparameter(model,'k_glc_tumor',params.k_glc_tumor.Value,'ValueUnits',params.k_glc_tumor.Units, 'Notes', 'Glucose consumption coefficient per cancer cell');
p.UserData.Source = params.k_glc_tumor.Notes;
    p.UserData.derivedFrom = params.k_glc_tumor.derivedFrom;

p = addparameter(model,'k_glc_immune',params.k_glc_immune.Value,'ValueUnits',params.k_glc_immune.Units, 'Notes', 'Glucose consumption coefficient per immune cell');
p.UserData.Source = params.k_glc_immune.Notes;
    p.UserData.derivedFrom = params.k_glc_immune.derivedFrom;

p = addparameter(model,'glc_50_Teff',params.glc_50_Teff.Value,'ValueUnits',params.glc_50_Teff.Units, 'Notes', 'Glucose level for half-max T cell effector function');
p.UserData.Source = params.glc_50_Teff.Notes;
    p.UserData.derivedFrom = params.glc_50_Teff.derivedFrom;

p = addparameter(model,'n_glc_Teff',params.n_glc_Teff.Value,'ValueUnits',params.n_glc_Teff.Units, 'Notes', 'Hill coefficient for glucose-T cell function');
p.UserData.Source = params.n_glc_Teff.Notes;
    p.UserData.derivedFrom = params.n_glc_Teff.derivedFrom;

p = addparameter(model,'k_lac_Csec',params.k_lac_Csec.Value,'ValueUnits',params.k_lac_Csec.Units, 'Notes', 'Lactate secretion rate per cancer cell (Warburg)');
p.UserData.Source = params.k_lac_Csec.Notes;
    p.UserData.derivedFrom = params.k_lac_Csec.derivedFrom;

p = addparameter(model,'k_lac_clear',params.k_lac_clear.Value,'ValueUnits',params.k_lac_clear.Units, 'Notes', 'Lactate clearance rate');
p.UserData.Source = params.k_lac_clear.Notes;
    p.UserData.derivedFrom = params.k_lac_clear.derivedFrom;

p = addparameter(model,'lac_50_M2',params.lac_50_M2.Value,'ValueUnits',params.lac_50_M2.Units, 'Notes', 'Lactate concentration for half-max M2 polarization drive');
p.UserData.Source = params.lac_50_M2.Notes;
    p.UserData.derivedFrom = params.lac_50_M2.derivedFrom;

p = addparameter(model,'lac_50_Teff',params.lac_50_Teff.Value,'ValueUnits',params.lac_50_Teff.Units, 'Notes', 'Lactate concentration for half-max T cell and NK suppression');
p.UserData.Source = params.lac_50_Teff.Notes;
    p.UserData.derivedFrom = params.lac_50_Teff.derivedFrom;

p = addparameter(model,'f_HIF_glycolysis',params.f_HIF_glycolysis.Value,'ValueUnits',params.f_HIF_glycolysis.Units, 'Notes', 'Fold-increase in glycolytic rate from HIF activation');
p.UserData.Source = params.f_HIF_glycolysis.Notes;
    p.UserData.derivedFrom = params.f_HIF_glycolysis.derivedFrom;

% Cancer cell glucose dependence (Warburg-adapted but still nutrient-limited)
% Lower EC50 than immune cells — cancer cells adapted to low glucose (GLUT1 overexpression)
glc_50_C = addparameter(model,'glc_50_C',params.glc_50_C.Value,...
    'ValueUnits',params.glc_50_C.Units,'ConstantValue',true,...
    'Notes',params.glc_50_C.Notes);
glc_50_C.UserData.Source = params.glc_50_C.Notes;
glc_50_C.UserData.derivedFrom = params.glc_50_C.derivedFrom;

H_glc_C = addparameter(model,'H_glc_C',1,'ValueUnits','dimensionless',...
    'ConstantValue',false,'Notes','Glucose limitation of cancer cell proliferation');

%% Add Algebraic Parameters (quasi-steady-state)
% Glucose equilibrates on minutes-to-hours timescale vs the model's daily timestep

glc = addparameter(model,'glc',0,'ValueUnits','micromolarity','ConstantValue',false, ...
    'Notes', 'Tumor tissue glucose concentration (micromolarity). Quasi-steady-state.');

%% Add Hill Functions (non-constant parameters, repeatedAssignment)

% Dimensionless ratio avoids SimBiology dimensional analysis error with ^n_glc_Teff
addparameter(model,'glc_ratio',0,'ValueUnits','dimensionless','ConstantValue',false, ...
    'Notes', 'glc/glc_50_Teff ratio (dimensionless). Intermediate for glucose Hill function.');

H_glc_Teff = addparameter(model,'H_glc_Teff',1,'ValueUnits','dimensionless','ConstantValue',false, ...
    'Notes', 'Glucose availability for T cell effector function (1=full, 0=impaired). Positive Hill.');

H_lac_M2 = addparameter(model,'H_lac_M2',0,'ValueUnits','dimensionless','ConstantValue',false, ...
    'Notes', 'Lactate-driven M2 polarization (0=none, 1=maximal).');

H_lac_Teff = addparameter(model,'H_lac_Teff',0,'ValueUnits','dimensionless','ConstantValue',false, ...
    'Notes', 'Lactate-driven suppression of T cell and NK effector function (0=none, 1=maximal).');

%% Add Rules for glucose, Hill functions
% Glucose: supply/demand balance following pO2 pattern
% Supply: proportional to K (vasculature), discounted by aberrant vasculature (VEGF)
% Demand: tumor cells (high) + immune/stromal cells (lower)
addrule(model, ['glc = glc_normal * V_T.K / (V_T.K + (k_glc_tumor * C_total' ...
    ' + k_glc_immune * (T_total + M_total + aPSC_total + V_T.MDSC + V_T.NK))' ...
    ' * (1 + V_T.VEGF / VEGF_norm_50))'], 'repeatedAssignment');

addrule(model, 'glc_ratio = glc / glc_50_Teff', 'repeatedAssignment');

% Positive Hill: 1 = full T cell function, 0 = impaired
addrule(model, 'H_glc_Teff = glc_ratio^n_glc_Teff / (1 + glc_ratio^n_glc_Teff)', 'repeatedAssignment');

% Lactate M2 polarization drive
addrule(model, 'H_lac_M2 = V_T.lactate / (lac_50_M2 + V_T.lactate)', 'repeatedAssignment');

% Lactate T cell and NK suppression (Brand 2016: 10-20 mM lactate impairs lymphocyte function)
addrule(model, 'H_lac_Teff = V_T.lactate / (lac_50_Teff + V_T.lactate)', 'repeatedAssignment');

% Glucose limitation of cancer cell proliferation (simple Hill, n=1)
addrule(model, 'H_glc_C = glc / (glc_50_C + glc)', 'repeatedAssignment');

%% Add ODE Species: Lactate
lactate = addspecies(model.Compartment(3),'lactate',0,'InitialAmountUnits','micromolarity');
    set(lactate,'Notes','Tumor lactate concentration from Warburg metabolism (micromolarity)');

%% Add Reactions: Lactate production and clearance
% Lactate production by cancer cells (Warburg effect, amplified by HIF)
% HIF-1alpha upregulates LDHA, HK2, GLUT1 -> increased glycolytic flux -> more lactate
reaction = addreaction(model,'null -> V_T.lactate');
    set(reaction,'ReactionRate','k_lac_Csec * C_total * glc / glc_normal * (1 + f_HIF_glycolysis * HIF)');
    set(reaction,'Notes','Lactate production by tumor cells (Warburg metabolism, HIF-amplified)');

% Lactate clearance (vascular washout + MCT-mediated export)
% Clearance scales with pO2/pO2_normal (vascular quality): as tumor grows
% and becomes more hypovascular, K/V_T drops, pO2 falls, and clearance
% decreases — allowing lactate to accumulate during progression.
% This replaces the earlier ad-hoc VEGF impairment factor (f_VEGF_lac),
% because pO2 already captures K vs demand balance including VEGF aberration.
% k_lac_clear is calibrated so effective clearance at diagnosis (~0.6/day)
% matches observed tumor lactate half-life of ~27 hours.
reaction = addreaction(model,'V_T.lactate -> null');
    set(reaction,'ReactionRate','k_lac_clear * V_T.lactate * pO2 / pO2_normal');
    set(reaction,'Notes','Lactate clearance from tumor (scales with vascular quality pO2/pO2_normal)');

%% Assembly-Time Modifications

% 1. Glucose impairment of T cell killing (k_C_Tcell_eff rule)
% Append * H_glc_Teff to the repeated assignment rule for effective killing rate
model_rules = get(model,'Rules');
for i = 1:length(model_rules)
    if ~isempty(strfind(strrep(model_rules(i).Rule,' ',''), 'k_C_Tcell_eff='))
        model_rules(i).Rule = [model_rules(i).Rule, ' * H_glc_Teff'];
    end
end

% 2-3. Glucose impairment of IFNg secretion (CD8 and Th)
for i = 1:length(model.reaction)
    if strcmp(model.reaction(i).reaction, 'null -> V_T.IFNg')
        rate = model.reaction(i).ReactionRate;
        if ~isempty(strfind(rate, 'k_IFNg_Tsec'))
            model.reaction(i).ReactionRate = [rate, '*H_glc_Teff'];
        elseif ~isempty(strfind(rate, 'k_IFNg_Thsec'))
            model.reaction(i).ReactionRate = [rate, '*H_glc_Teff'];
        end
    end
end

% 4. Lactate-driven M2 macrophage polarization
% Modify M1->M2 rate: replace *(1-HIF)) with *(1-HIF)*(1-H_lac_M2))
for i = 1:length(model.reaction)
    if strcmp(model.reaction(i).reaction, 'V_T.Mac_M1 -> V_T.Mac_M2')
        oldRate = model.reaction(i).ReactionRate;
        newRate = strrep(oldRate, '*(1-HIF))', '*(1-HIF)*(1-H_lac_M2))');
        if ~strcmp(oldRate, newRate)
            model.reaction(i).ReactionRate = newRate;
        end
    end
end

% 5. Lactate suppression of T cell killing (k_C_Tcell_eff rule)
% Direct pH-mediated impairment of TCR signaling and cytolytic function (Brand 2016)
for i = 1:length(model_rules)
    if ~isempty(strfind(strrep(model_rules(i).Rule,' ',''), 'k_C_Tcell_eff='))
        model_rules(i).Rule = [model_rules(i).Rule, ' * (1-H_lac_Teff)'];
    end
end

% 6-7. Lactate suppression of IFNg secretion (CD8, Th)
% Appended alongside existing H_glc_Teff modifications
for i = 1:length(model.reaction)
    if strcmp(model.reaction(i).reaction, 'null -> V_T.IFNg')
        rate = model.reaction(i).ReactionRate;
        if ~isempty(strfind(rate, 'k_IFNg_Tsec'))
            model.reaction(i).ReactionRate = [rate, '*(1-H_lac_Teff)'];
        elseif ~isempty(strfind(rate, 'k_IFNg_Thsec'))
            model.reaction(i).ReactionRate = [rate, '*(1-H_lac_Teff)'];
        end
    end
end

% 8-9. Lactate suppression of NK cell function (killing and IFNg)
% Lactate inhibits all lymphocyte subtypes (Brand 2016)
for i = 1:length(model.reaction)
    rate = model.reaction(i).ReactionRate;
    if ~isempty(strfind(rate, 'k_NK_kill'))
        model.reaction(i).ReactionRate = [rate, '*(1-H_lac_Teff)'];
    elseif ~isempty(strfind(rate, 'k_IFNg_NK_sec'))
        model.reaction(i).ReactionRate = [rate, '*(1-H_lac_Teff)'];
    end
end

% 10. Glucose limitation of cancer cell proliferation
% Append * H_glc_C to cancer growth reaction
for i = 1:length(model.reaction)
    rate = model.reaction(i).ReactionRate;
    if contains(rate, 'k_C1_growth') && ~contains(rate, 'H_glc_C')
        model.reaction(i).ReactionRate = [rate, ' * H_glc_C'];
    end
end

end
