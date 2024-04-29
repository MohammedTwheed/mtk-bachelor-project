% 
% clear; clc; clf;
% Load data
% load('QH.mat');
% load('D.mat');

% 

input  = QH';
target = D';

% [a,b,c,d]=optimizeNNForTrimmingPumpImpeller(QH',D');
[a,b,c,d,e]=optimizeNNForTrimmingPumpImpeller2(QH',D');

% Test the Network
y = d(input);



% now the problem was with the number of data points
% we need just to understand the dimensionality of 
% inputs and outputs
% QH was 331x2 so its transpose is 2x331.
% D was 331x1 so its transpose is 1x331.
% this means that the network takes in 2 inputs and get 1
% output for a 331 points.

Q = QH(:,1);
H = QH(:,2);
ND = sim(d,QH');

[Qq, Hq] = meshgrid(0:2:440, 0:.5:90);
Dq = griddata(Q, H, ND, Qq, Hq);

figure;
mesh(Qq, Hq, Dq);
hold on;
scatter3(Q, H, D, 'o', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
xlabel('Flow Rate (m^3/h)');
ylabel('Head (m)');
zlabel('Diameter (mm)');
title('Fitted Function and Data Points');
legend('Fitted Function', 'Data Points');

h = gca;
h.XLim = [0 440];
h.YLim = [0 90];


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
    % to find the optimum architecture for example call optimizeNNForTrimmingPumpImpeller
    % [a,b,c,d,probs_out]=optimizeNNForTrimmingPumpImpeller(QH_nn_input',D_eta_nn_output')
    % probs_out is to use with mapminmax('reverse',bestTrainedNet(your
    % input),probs_out) to get scale back since we have normalized it.
    %please make sure you do the transpose if you are using our 'trimming_nn_training_dataset.mat'


    % Start timer to measure the duration of the optimization process.
    tic;
    disp("Optimization exploration_02 in progress. This process may take up to 30 seconds...");

    % Define the available options for training functions and activation functions.
    trainingFunctionOptions = {'trainlm', 'trainbr', 'trainrp', ...
        'traincgb', 'traincgf', 'traincgp', 'traingdx', 'trainoss'};
    activationFunctionOptions = {'tansig', 'logsig'};

    % Define bounds for the genetic algorithm optimization.
    % the positional meaning [<hidden layer neurons number> ,< epochs>...
    % ,<index of trainingFunctionOptions>,<index of activationFunctionOptions>]
    lowerBounds = [5,  5,    50, 1, 1];
    upperBounds = [200,200, 200, 8, 2];

    % Define options for the genetic algorithm.
    % ConstraintTolerance, is the convergence limit its value determines
    % the stop criteria for the ga.
    % FitnessLimit, is the min mse value to stop search if mse get to it.
    gaOptions = optimoptions('ga', 'MaxTime', 20,'ConstraintTolerance',0.0003,'FitnessLimit',0.0009);
    % gaOptions = optimoptions('ga', 'MaxTime', 2);

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


    % local function to evaluate hyperparameters using the neural network.
    function avgMSEs = evaluateHyperparameters(hyperParams, x, t, randomSeed)
        rng(randomSeed); % Set random seed for reproducibility.

        % Extract hyperparameters.
        hiddenLayer1Size = round(hyperParams(1)); %Hidden Layer Size
        hiddenLayer2Size = round(hyperParams(2)); %Hidden Layer Size
        maxEpochs = round(hyperParams(3));       %Max Epochs
        trainingFunctionIdx = round(hyperParams(4)); %Training Function
        activationFunctionIdx = round(hyperParams(5));%Activation Function or transfere function

        % Define the neural network.
        net = feedforwardnet([hiddenLayer1Size,hiddenLayer2Size],...
            trainingFunctionOptions{trainingFunctionIdx});
        % Suppress training GUI for efficiency.
        net.trainParam.showWindow = false; 
        net.trainParam.epochs = maxEpochs;
        net.layers{1}.transferFcn = activationFunctionOptions{activationFunctionIdx};
        net.layers{2}.transferFcn = activationFunctionOptions{activationFunctionIdx};

        % Choose a Performance Function
        net.performFcn = 'mse';


        % Choose Input and Output Pre/Post-Processing Functions
        net.input.processFcns = {'removeconstantrows', 'mapminmax'};
        net.output.processFcns = {'removeconstantrows', 'mapminmax'};

        % Define data split for training, validation, and testing.

        % For a list of all data division functions type: help nndivide
        net.divideFcn = 'dividerand';  % Divide data randomly
        net.divideMode = 'sample';  % Divide up every sample
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.15;
        net.divideParam.testRatio = 0.15;

        % Train the neural network.
        [trainedNet, tr] = train(net, x, t);

        % Evaluate the model performance using mean squared error (MSE).

        % predictions = trainedNet(normalized_input);
        predictions = trainedNet(x);
        mse = perform(trainedNet, t, predictions);

        % Recalculate Training, Validation and Test Performance
        trainTargets        = t .* tr.trainMask{1};
        valTargets          = t .* tr.valMask{1};
        testTargets         = t .* tr.testMask{1};
        trainPerformance    = perform(net,trainTargets,predictions);
        valPerformance      = perform(net,valTargets,predictions);
        testPerformance     = perform(net,testTargets,predictions);

        % for better performance we came up with this convention
        % rather than using the mse based on perform function only
        avgMSEs = (mse +  ...
            trainPerformance +...
            valPerformance+....
            testPerformance) / 4;

    
        % Check if the current MSE is the best MSE so far and update the global variable if necessary.
        if isempty(bestTrainedNet) || avgMSEs < perform(bestTrainedNet, t, bestTrainedNet(x))
            bestTrainedNet = trainedNet;
        end
    end

    % Set a random seed for reproducibility.
    randomSeed = randi(10000);
    rng(randomSeed);

    % Perform optimization using genetic algorithm.
    [optimalHyperParams, finalMSE] = ga(@(hyperParams) evaluateHyperparameters(hyperParams, x, t, randomSeed), ...
        5, [], [], [], [], lowerBounds, upperBounds, [], gaOptions);

    % Round the optimized hyperparameters to integers.
    optimalHyperParams = round(optimalHyperParams);

    % Measure elapsed time.
    elapsedTime = toc;

    % Extract the chosen training and activation functions.
    trainingFunction = trainingFunctionOptions{optimalHyperParams(4)};
    activationFunction = activationFunctionOptions{optimalHyperParams(5)};

    % Display the optimization results.
    fprintf('Optimized Hyperparameters: Hidden Layer 1 Size = %d, Hidden Layer 2 Size = %d,Max Epochs = %d, Training Function = %s, Activation Function = %s\n', ...
        optimalHyperParams(1), optimalHyperParams(2),optimalHyperParams(3), trainingFunction, activationFunction);
    fprintf('Final Mean Squared Error (MSE): %.4f\n', finalMSE);
    fprintf('Random Seed Used: %d\n', randomSeed);
    fprintf('Optimization Duration: %.4f seconds\n', elapsedTime);

 
end