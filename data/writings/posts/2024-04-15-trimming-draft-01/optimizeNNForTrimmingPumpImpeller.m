function [optimalHyperParams, finalMSE, randomSeed, bestTrainedNet] = optimizeNNForTrimmingPumpImpeller(x, t)
    % Function to optimize neural network hyperparameters for the trimming of a pump impeller.
    % Inputs:
    %   x - Input data (Q flowrate,H head) for the neural network.
    %   t - Target data (D diameter,eta efficiency) for the neural network.
    % Outputs:
    %   optimalHyperParams - Optimized hyperparameters found by the genetic algorithm.
    %   finalMSE - Mean squared error (MSE) of the best model.
    %   randomSeed - Random seed used for reproducibility.
    %   bestTrainedNet - The best trained neural network found during optimization.

    %EXAMPLE USAGE
    % first load the dataset using load('trimming_nn_training_dataset.mat')
    % to find the optimum architecture for example call optimizeNNForTrimmingPumpImpeller(QH_nn_input',D_eta_nn_output')
    %please make sure you do the transpose 

    % Start timer to measure the duration of the optimization process.
    tic;
    disp("Optimization in progress. This process may take up to 30 seconds...");

    % Define the available options for training functions and activation functions.
    trainingFunctionOptions = {'trainlm', 'trainbr', 'trainrp', 'traincgb', 'traincgf', 'traincgp', 'traingdx', 'trainoss'};
    activationFunctionOptions = {'tansig', 'logsig'};

    % Define bounds for the genetic algorithm optimization.
    lowerBounds = [5, 50, 1, 1];
    upperBounds = [200, 200, 8, 2];

    % Define options for the genetic algorithm.
    gaOptions = optimoptions('ga', 'MaxTime', 20);

    % Global variable to store the best trained neural network found during optimization.
    global bestTrainedNet;
    bestTrainedNet = [];
    % TODO: MTK to SEI you might consider making a helper function 
    % just to resolve this issue with the ga for example 
    % function  [mse,bestTrainedNet] = evaluateHyperparameters(params...)
    %  end
    % function  mse = f_ga(params...) 
    % [mse,bestTrainedNet] = evaluateHyperparameters(params...)
    %  end
    % but would this leak the bestTrainedNet #NeedResearch ??!!
    % monadic approach to side effects as in haskell
    % SEI: just re train it MTK:!!


    % Nested function to evaluate hyperparameters using the neural network.
    function mse = evaluateHyperparameters(hyperParams, x, t, randomSeed)
        rng(randomSeed); % Set random seed for reproducibility.

        % Extract hyperparameters.
        hiddenLayerSize = round(hyperParams(1)); %Hidden Layer Size
        maxEpochs = round(hyperParams(2));       %Max Epochs
        trainingFunctionIdx = round(hyperParams(3)); %Training Function
        activationFunctionIdx = round(hyperParams(4));%Activation Function or transfere function

        % Define the neural network.
        net = fitnet(hiddenLayerSize, trainingFunctionOptions{trainingFunctionIdx});
        net.trainParam.showWindow = false; % Suppress training GUI for efficiency.
        net.trainParam.epochs = maxEpochs;
        net.layers{1}.transferFcn = activationFunctionOptions{activationFunctionIdx};

        % Define data split for training, validation, and testing.
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.15;
        net.divideParam.testRatio = 0.15;

        % Train the neural network.
        [trainedNet, ~] = train(net, x, t);

        % Evaluate the model performance using mean squared error (MSE).
        predictions = trainedNet(x);
        mse = perform(trainedNet, t, predictions);

        % Check if the current MSE is the best MSE so far and update the global variable if necessary.
        if isempty(bestTrainedNet) || mse < perform(bestTrainedNet, t, bestTrainedNet(x))
            bestTrainedNet = trainedNet;
        end
    end

    % Set a random seed for reproducibility.
    randomSeed = randi(10000);
    rng(randomSeed);

    % Perform optimization using genetic algorithm.
    [optimalHyperParams, finalMSE] = ga(@(hyperParams) evaluateHyperparameters(hyperParams, x, t, randomSeed), ...
        4, [], [], [], [], lowerBounds, upperBounds, [], gaOptions);

    % Round the optimized hyperparameters to integers.
    optimalHyperParams = round(optimalHyperParams);

    % Measure elapsed time.
    elapsedTime = toc;

    % Extract the chosen training and activation functions.
    trainingFunction = trainingFunctionOptions{optimalHyperParams(3)};
    activationFunction = activationFunctionOptions{optimalHyperParams(4)};

    % Display the optimization results.
    fprintf('Optimized Hyperparameters: Hidden Layer Size = %d, Max Epochs = %d, Training Function = %s, Activation Function = %s\n', ...
        optimalHyperParams(1), optimalHyperParams(2), trainingFunction, activationFunction);
    fprintf('Final Mean Squared Error (MSE): %.4f\n', finalMSE);
    fprintf('Random Seed Used: %d\n', randomSeed);
    fprintf('Optimization Duration: %.4f seconds\n', elapsedTime);

 
end
