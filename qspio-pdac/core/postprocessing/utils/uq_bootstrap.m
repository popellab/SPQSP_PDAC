load('QSPIO-HCC/simulation_results/nivo/simulation_result.mat')
% Initialize parameters
sample_size = 100;
iteration = 200;

% Storage for the count of CR/PR patients in each iteration
bootstrap_perc_CR_PR = zeros(iteration, 1);

bootstrap_trajec_PD = [];
bootstrap_trajec_SD = [];
bootstrap_trajec_CR = [];

% Perform bootstrapping
for i = 1:iteration
    % Bootstrapped sample without replacement
    bootstrapped_sample = randsample(params_out.iPatientPlaus, sample_size, false);

    % Initialize the CR/PR count for this iteration
    count_CR_PR = 0;
    trajec_PD_temp = [];
    trajec_SD_temp = [];
    trajec_CR_temp = [];

    % Loop over the bootstrapped samples
    for j = 1:sample_size
        patient_index = bootstrapped_sample(j);

        % Find index of D_T_perc
        D_T_index = find(strcmp(simDataPSApost(patient_index).simData.DataNames, 'D_T_perc'));
        D_T_perc = simDataPSApost(patient_index).simData.Data(:, D_T_index);

        % Get the RECIST category for this patient
        recist_category = params_out.RECIST{patient_index};

        % Count if the RECIST category is 'CR/PR'
        if ~isempty(D_T_perc)
            if strcmp(recist_category, 'CR/PR')
                count_CR_PR = count_CR_PR + 1;
                trajec_CR_temp = [trajec_CR_temp D_T_perc];
            elseif strcmp(recist_category, 'SD')
                trajec_SD_temp = [trajec_SD_temp D_T_perc];
            elseif strcmp(recist_category, 'PD')
                trajec_PD_temp = [trajec_PD_temp D_T_perc];
            end
        end
    end

    % Store the mean trajectories
    bootstrap_trajec_PD = [bootstrap_trajec_PD mean(trajec_PD_temp, 2)];
    bootstrap_trajec_SD = [bootstrap_trajec_SD mean(trajec_SD_temp, 2)];
    bootstrap_trajec_CR = [bootstrap_trajec_CR mean(trajec_CR_temp, 2)];

    % Store the percentage of CR/PR patients for this iteration
    bootstrap_perc_CR_PR(i) = count_CR_PR / sample_size;
end

% Calculate the mean and 95% confidence intervals for CR/PR patients
mean_CR_PR = mean(bootstrap_perc_CR_PR, 1);
ci_CR_PR_lower = prctile(bootstrap_perc_CR_PR, 2.5, 1);
ci_CR_PR_upper = prctile(bootstrap_perc_CR_PR, 97.5, 1);

result_string = sprintf('Mean response rate: %.3f, 95%% confidence interval: (%.3f, %.3f)\n', mean_CR_PR, ci_CR_PR_lower, ci_CR_PR_upper);
disp(result_string)
% Plotting CR trajectories
n_time_points = size(bootstrap_trajec_CR, 1);

% Calculate mean and 95% confidence intervals across iterations for each time point
mean_CR = mean(bootstrap_trajec_CR, 2);
ci_lower_CR_traj = prctile(bootstrap_trajec_CR, 2.5, 2)';
ci_upper_CR_traj = prctile(bootstrap_trajec_CR, 97.5, 2)';

%for PD
mean_PD = mean(bootstrap_trajec_PD, 2);
ci_lower_PD_traj = prctile(bootstrap_trajec_PD, 2.5, 2)';
ci_upper_PD_traj = prctile(bootstrap_trajec_PD, 97.5, 2)';

%for SD
mean_SD = mean(bootstrap_trajec_SD, 2);
ci_lower_SD_traj = prctile(bootstrap_trajec_SD, 2.5, 2)';
ci_upper_SD_traj = prctile(bootstrap_trajec_SD, 97.5, 2)';

% Plot the mean and the 95% confidence intervals
time_points = 1:n_time_points;
time_points = time_points/10;
figure;
hold on;
h1 = plot(time_points, mean_CR, 'Color', [0 0.4470 0.7410], 'LineWidth', 2);
fill([time_points, fliplr(time_points)], [ci_lower_CR_traj, fliplr(ci_upper_CR_traj)], [0 0.4470 0.7410], 'FaceAlpha', 0.2, 'EdgeColor', 'none');

h2 = plot(time_points, mean_PD, 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2);
fill([time_points, fliplr(time_points)], [ci_lower_PD_traj, fliplr(ci_upper_PD_traj)], [0.8500 0.3250 0.0980], 'FaceAlpha', 0.2, 'EdgeColor', 'none');

h3 = plot(time_points, mean_SD, 'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2);
fill([time_points, fliplr(time_points)], [ci_lower_SD_traj, fliplr(ci_upper_SD_traj)], [0.9290 0.6940 0.1250], 'FaceAlpha', 0.2, 'EdgeColor', 'none');

xlabel('Time (Days)');
ylabel('Tumor Volume Change (%)');
xlim([0 time_points(end)]);
legend([h1, h2, h3], 'CR/PR', 'PD', 'SD');
hold off;

%% Plot Results
% n_PSA = length(params_out.RECIST(params_out.iPlausSim));
n_PSA = length(params_out.iPatientPlaus);
index = params_out.iPatientPlaus;

time = 0:0.1:400;
time = time./30;

% n_PSA = length(params_out.RECIST(params_out.iPlausSim));
n_PSA = length(params_out.iPatientPlaus);
index = params_out.iPatientPlaus;

figure; hold on; box on;
responders = 0;
j = [find(strcmp(simDataPSApost(index(1)).simData.DataNames,'D_T_perc') )];

for i =1:n_PSA
    plot(time, ...
         simDataPSApost(index(i)).simData.Data(:,j) ,'DisplayName',num2str(index(i)), 'LineWidth', 2)
    D_T_perc = simDataPSApost(index(i)).simData.Data(:,j);
    if(min(D_T_perc(57:56:end)) < -30)
        responders = responders + 1;
    end
end
fprintf('ORR rate = %f\n',responders/n_PSA);
plot([0 max(time)*1.25], [+20 +20] , '--k' )
plot([0 max(time)*1.25], [-30 -30] , '--k' )
hx = text(max(time)*1.10, 60,'PD');
hx = text(max(time)*1.10,  0,'SD');
hx = text(max(time)*1.05,-60,'PR/CR');
xlabel('Time (month)','Fontsize',12);
ylabel('Percent tumor size change (\%)','Fontsize',12);
ylim([-100 100])
xlim([0  max(time)*1.25])
set(gca,'Fontsize',14)
