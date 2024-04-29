clear;clc;clf;

% Define the neural network.

% Load data
load('QH.mat');
load('D.mat');



% Define input and target data
inputs = QH';
targets = D';

        net = feedforwardnet([62], 'trainlm');
        net.trainParam.showWindow = false; % Suppress training GUI for efficiency.

        % % Choose a Performance Function
        % net.performFcn = 'mse';


        % % Choose Input and Output Pre/Post-Processing Functions
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
        [trainedNet, ~] = train(net, inputs, targets);

Q = QH(:,1);
H = QH(:,2);
ND = sim(trainedNet,QH');

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
