% SimBiology Plot tumor size for PSA cases and RECIST criteria
%
% Inputs: simDataPSA     -- Object containing the simbiology model outputs
%                            for all batch simulations
%         simDataPSApost -- Object containing the post processed simbiology
%                           model outputs for all batch simulations
%         params_out     -- object containing model outputs with RCEIST and
%                           Response status


function PSA_plot_TumSize(simDataPSA,simDataPSApost,params_out)

% n_PSA = length(params_out.RECIST(params_out.iPlausSim));
n_PSA = length(params_out.iPatientPlaus);
index = params_out.iPatientPlaus;

figure; hold on; box on;

time = simDataPSA(index(1)).simData.Time/30;
j1 = [find(strcmp(simDataPSApost(index(1)).simData.DataNames,'D_T_perc') )];

for i =1:n_PSA
    plot(simDataPSA(index(i)).simData.Time/30, ...
         simDataPSApost(index(i)).simData.Data(:,j1) ,'DisplayName',num2str(index(i)), 'LineWidth', 2)
end
xlabel('Time (month)','Fontsize',12);
ylabel('Tumor size (mm)','Fontsize',12);
% ylim([-100 150])
xlim([0  max(time)])
set(gca,'Fontsize',14)


%% cancer cell count
%{
figure; hold on; box on;

time = simDataPSA(index(1)).simData.Time/30;
j2 = [find(strcmp(simDataPSA(index(1)).simData.DataNames,'C1') )];
j3 = [find(strcmp(simDataPSA(index(1)).simData.DataNames,'C2') )];
for i =1:n_PSA
    plot(simDataPSA(index(i)).simData.Time/30, ...
         simDataPSA(index(i)).simData.Data(:,j2) + simDataPSA(index(i)).simData.Data(:,j3) ,'DisplayName',num2str(index(i)), 'LineWidth', 2)
end
xlabel('Time (month)','Fontsize',12);
ylabel('Cancer counts','Fontsize',12);
% ylim([-100 150])
xlim([0  max(time)])
set(gca,'Fontsize',14)
%}
