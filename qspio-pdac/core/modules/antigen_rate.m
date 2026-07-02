% Function to generate object with default model parameters
%
% Inputs: model   -- SimBiology model object
%         ID      -- T cell-antigen ID number
%         cancers -- cell array of string containing cancer names
%         P_C     -- parameter array (same length as 'cancers') of antigen concentrations per cancer cell
%         nTcells -- number of cytotoxic T cell clones in model
%
% Output: rate    -- string containing rate of antigen release from dying cancer cells


function rate = antigen_rate(model,ID,cancers,P_C,nTcells)

% Parse Inputs
P = ['P' ID];
C = cancers;

% Add Antigen Concentrations to Model
for i = 1:length(P_C)
    param = addparameter(model,[P '_' C{i}],P_C(i).Value,'ValueUnits',P_C(i).Units, 'Notes', ['Concentration of ' P ' in ' C{i}]);
    param.UserData.Source = P_C(i).Notes;
end

rate = '';
for i = 1:length(C)
    R_T = '0*cell/day'; % default value if no T cell killing reaction found
    for k = 1:length(model.reaction)
        if strcmp(model.reaction(k).reaction, ['V_T.' C{i} ' -> V_T.C_x']) ...
            && ~isempty(strfind(model.reaction(k).ReactionRate, 'k_C_Tcell_eff'))
            R_T = model.reaction(k).ReactionRate;
        end
    end
    if strcmp(rate,'')
        % rate = rate + P_Ci*(k_Ci_death+k_Ci_therapy)*Ci + P_Ci*R_Ti
        rate = sprintf('%s_%s*(k_%s_death+k_%s_therapy)*%s+%s_%s*%s',...
            P,C{i},C{i},C{i},C{i},P,C{i},R_T);
    else
        rate = sprintf('%s+%s_%s*(k_%s_death+k_%s_therapy)*%s+%s_%s*%s',...
            rate,P,C{i},C{i},C{i},C{i},P,C{i},R_T);
    end
end
% Map antigen ID to the new parameter naming convention
if strcmp(ID, '0')
    n_clones_param = 'n_CD4_clones';  % self-antigen clone diversity (shared CD4 pool)
elseif strcmp(ID, '1')  
    n_clones_param = 'n_CD8_clones';  % tumor antigens (renamed from n_clones_tum)
else
    error('Unknown antigen ID: %s. Expected ''0'' (self) or ''1'' (tumor)', ID);
end
rate = [n_clones_param '*(' rate ')'];
