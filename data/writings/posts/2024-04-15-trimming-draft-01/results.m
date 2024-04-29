% % Load dataset
% load('trimming_nn_training_dataset.mat');
% 
% % Initialize variables to store results
% result = zeros(10, 7); % 10 iterations, 4 hyperparameters, final MSE, random seed
% 
% for i = 1:1
%     % Call optimization function
%     [optimalHyperParams, finalMSE, randomSeed, bestTrainedNet,probs_out] = optimizeNNForTrimmingPumpImpeller(QH_nn_input', D_eta_nn_output');
% 
%     % Store results
%     result(i, :) = [i, optimalHyperParams, finalMSE, randomSeed];
% 
%     % Plot bestTrainedNet vs training data
%     figure;
%     plot(QH_nn_input', D_eta_nn_output', 'bo', 'DisplayName', 'Training Data');
%     hold on;
%     [normalized_input,probs_in]=mapminmax(QH_nn_input');
%     predictions = bestTrainedNet(normalized_input);
%     predictions = mapminmax('reverse', predictions,probs_out);
%     plot(mapminmax('reverse', normalized_input,probs_out), predictions, 'r-', 'LineWidth', 2, 'DisplayName', 'Predictions');
%     legend('show');
%     xlabel('QH\_nn\_input');
%     ylabel('D\_eta\_nn\_output');
%     title(['Iteration ', num2str(i), ': BestTrainedNet vs Training Data']);
%     grid on;
% 
%     % Save plot
%     plotFileName = [num2str(i), '_', num2str(optimalHyperParams(1)), '-', num2str(optimalHyperParams(2)), '-', num2str(optimalHyperParams(3)), '-', num2str(optimalHyperParams(4)), '.png'];
%     saveas(gcf, plotFileName);
%     close(gcf);
% end
% 
% % Write resultto CSV file
% writematrix(result, 'optimization_results.csv');





% Load the training data
load('trimming_nn_training_dataset.mat');

% Define the number of iterations for loop
numIterations = 1;

% Initialize an empty array to store results
result= zeros(numIterations, 7);

for i = 1:numIterations
  % Perform optimization
  [optimalHyperParams, finalMSE, randomSeed, bestTrainedNet, probs_out] = optimizeNNForTrimmingPumpImpeller(QH_nn_input', D_eta_nn_output');

  % Store resultfor this iteration
  result(i,:) = [i, optimalHyperParams, finalMSE, randomSeed];

  % Plot result(Q,H) vs D and (Q,H) vs eta
  figure;

  % First predicted output (D)
  subplot(2,2,1);
  x = QH_nn_input';
  scatter3(x(:,1)', x(:,2)', D_eta_nn_output(:,1));
  hold on;
  predictions = bestTrainedNet(mapminmax('process', x));
  predictions = mapminmax('reverse', predictions(:,1), probs_out(:,1));
  scatter3(x(:,1)', x(:,2)', predictions);
  xlabel('Q (m^3/h)');
  ylabel('H (m)');
  zlabel('D (mm)');
  title('(Q,H) vs Predicted D');

  % Second predicted output (eta)
  subplot(2,2,2);
  scatter3(x(:,1)', x(:,2)', D_eta_nn_output(:,2));
  hold on;
  predictions = bestTrainedNet(mapminmax('process', x));
  predictions = mapminmax('reverse', predictions(:,2), probs_out(:,2));
  scatter3(x(:,1), x(:,2), predictions);
  xlabel('Q (m^3/h)');
  ylabel('H (m)');
  zlabel('eta');
  title('(Q,H) vs Predicted eta');

  % Save the plot with a descriptive filename
  filename = sprintf('%d_%d-%d-%d-%d.png', i, optimalHyperParams(1), optimalHyperParams(2), optimalHyperParams(3), optimalHyperParams(4));
  saveas(gcf, filename);

  % Close the figure to avoid memory issues
  close(gcf);
end

% Write the resultto a CSV file
writematrix([["Iteration", "Hidden Layer Size", "Max Epochs", "Training Function", "Activation Function", "Final MSE", "Random Seed"]; results], 'results.csv');

disp('resultsaved to results.csv');