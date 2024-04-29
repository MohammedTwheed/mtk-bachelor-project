clear;clc;clf;
load('trimming_nn_training_dataset.mat');
% [a,b,c,d,e]=optimizeNNForTrimmingPumpImpeller2(QH_nn_input',D_eta_nn_output');

[a,b,c,d]=optimizeNNForTrimmingPumpImpeller2(QH_nn_input',D_eta_nn_output');

% % Define the number of points
% num_points = 200;
% 
% % Define the ranges for X and Y
% x_range = linspace(150, 400, sqrt(num_points));
% y_range = linspace(0, 90, sqrt(num_points));
% 
% 
% 
% predic = d(mapminmax([x_range;y_range]));
% predic_t = predic';
% predic_rn = mapminmax('reverse',predic,e);
% predic_rn_t = predic_rn';
% 
% predic_1 = d(mapminmax(QH_nn_input'));
% predic_1_rn = mapminmax('reverse',predic_1,e);
%  predic_1_rn_t=predic_1_rn';
% % scatter3(QH_nn_input(:,1), QH_nn_input(:,2), predic_1_rn_t(:,1),'r', 'filled')
% scatter3(QH_nn_input(:,1), QH_nn_input(:,2), D_eta_nn_output(:,1),'r', 'filled')
% hold on
% [Q, H] = meshgrid(x_range, y_range);
% D= meshgrid(predic_rn_t(:,1));
% surf(Q, H,D,'FaceAlpha', 0.5)
% 
% xlabel('Q');
% ylabel('H');
% zlabel('D');
% 
% % Add legend
% legend('Scatter Plot', 'Surface Plot');
% 
% hold off; % Release the hold on the plot


% Define the number of points
num_points = 200;

% Define the ranges for X and Y
x_range = linspace(150, 400, sqrt(num_points));
y_range = linspace(0, 90, sqrt(num_points));



predic = d([x_range;y_range]);


% predic_1 = d(QH_nn_input');

% scatter3(QH_nn_input(:,1), QH_nn_input(:,2), predic_1_rn_t(:,1),'r', 'filled')
scatter3(QH_nn_input(:,1), QH_nn_input(:,2), D_eta_nn_output(:,1),'r', 'filled')
hold on
[Q, H] = meshgrid(x_range, y_range);
predic=predic';
D= meshgrid(predic(:,1));
surf(Q, H,D,'FaceAlpha', 0.5)

xlabel('Q');
ylabel('H');
zlabel('D');

% Add legend
legend('Scatter Plot', 'Surface Plot');

hold off; % Release the hold on the plot