%Autoregression Time-Series with a NAR Neural Network
% First open output_n.mat from the folder
% output1_n is the normalized output of the preprocess phase
T = tonndata(output1_n,false,false);
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

% The Nonlinear Autoregressive Network
feedbackDelays = 1:24;
hiddenLayerSize = 10;
net = narnet(feedbackDelays,hiddenLayerSize,'open',trainFcn);

% Choose Feedback Pre/Post-Processing Functions
net.input.processFcns = {'removeconstantrows','mapminmax'};

% Prepare the Data for Training and Simulation
[x,xi,ai,t] = preparets(net,{},{},T);
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'time';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.performFcn = 'mse';  % Mean Squared Error

% Plot Functions
net.plotFcns = {'plotperform','plottrainstate', 'ploterrhist', ...
    'plotregression', 'plotresponse', 'ploterrcorr', 'plotinerrcorr'};
[net,tr] = train(net,x,t,xi,ai);% Train the Network

y = net(x,xi,ai);% Test the Network
e = gsubtract(t,y);
performance = perform(net,t,y)

% Recalculate Training, Validation and Test Performance
trainTargets = gmultiply(t,tr.trainMask);
valTargets = gmultiply(t,tr.valMask);
testTargets = gmultiply(t,tr.testMask);
trainPerformance = perform(net,trainTargets,y)
valPerformance = perform(net,valTargets,y)
testPerformance = perform(net,testTargets,y)

view(net)% View the Network
