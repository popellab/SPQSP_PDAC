% Produce waterfall plot for percent change in tumor size for a parameter
%
% Inputs: simDataPSApost -- Object containing the post processed simbiology
%                           model outputs for all batch simulations
%         params_in      -- Object containing the input parameters
%         param          -- name of the parameter used for color coding


function PSA_box_plot(simDataPSApost,model,params_in,params_out,treated)

n_PSA = length(params_out.iPatientPlaus);
index = params_out.iPatientPlaus;
j = [find(strcmp(simDataPSApost(index(1)).simData.DataNames,'D_T_perc') )];
for i =1:n_PSA
    D_T_perc = simDataPSApost(index(i)).simData.Data(:,j);
    % temp(i,1) = D_T_perc(end); % end tumor size
    % best overall response assuming tumor measurement every 8 weeks
    tumor_change_perc(i,1) = min(D_T_perc(57:56:end));
end

f = figure;
% % Set the window in a specific size
% set(f0100,'Position', [50 50 1300 800]);
% Maximized window
set(f,'units','normalized','outerposition',[0 0 1 1])
for i=1:length(simDataPSApost(index(1)).simData.DataNames)
    subplot(5,5,i); hold on; box on;
    var = [];
    response = {};
    for j=1:n_PSA
        if tumor_change_perc(j, 1) >= -30
            response{j,1} = 'NR';
        else
            response{j,1} = 'R';
        end

        if treated == 1
            var(j,1) = simDataPSApost(index(j)).simData.Data(end,i);
        else
            var(j,1) = simDataPSApost(index(j)).simData.Data(1,i);
        end
    end
    boxplot(var,response)
    xlabel('Response')
    ylabel(simDataPSApost(index(1)).simData.DataNames(i))

    % Extract data for the two groups
    group_NR = var(strcmp(response, 'NR'));
    group_R = var(strcmp(response, 'R'));

    % Perform a t-test to compare the two groups
    [h, p] = ttest2(group_NR, group_R);

    % Annotate the boxplot with the p-value
    y_max = max(var);
    y_offset = 0.05 * range(var); % Offset for the annotation above the max value

    text(1.5, y_max + y_offset, ['p = ' num2str(p)], 'HorizontalAlignment', 'center', 'FontSize', 10);

    % Draw a line connecting the two boxplots
    plot([1, 2], [y_max + y_offset * 0.8, y_max + y_offset * 0.8], '-k', 'LineWidth', 1);
end
