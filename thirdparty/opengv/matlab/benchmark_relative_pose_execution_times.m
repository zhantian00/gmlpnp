%% Reset everything

clear all;
clc;
close all;
addpath('helpers');

%% Configure the benchmark

% central case -> only one camera
cam_number = 1;
% Getting 10 points, and testing all algorithms with the respective number of points
pt_number = 10;
% noise test, so no outliers
outlier_fraction = 0.0;
% repeat 1000 iterations
iterations = 1000;

% The algorithms we want to test
algorithms = { 'twopt';'fivept_stewenius'; 'fivept_nister'; 'fivept_kneip'; 'sevenpt'; 'eightpt'; 'eigensolver' ; 'sixpt_urban_onlyRot'};
% This defines the number of points used for every algorithm
indices = { [1,2]; [1, 2, 3, 4, 5]; [1, 2, 3, 4, 5];  [1, 2, 3, 4, 5]; [1, 2, 3, 4, 5, 6, 7]; [1, 2, 3, 4, 5, 6, 7, 8]; [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];[1:10] };
% The name of the algorithms in the final plots
names = { '2pt';'5pt (Stewenius)'; '5pt (Nister)'; '5pt (Kneip)'; '7pt'; '8pt'; 'eigensolver'; 'sixpt_urban' };

% noise in this experiment
noise = 1.0;

%% Run the benchmark

%prepare the overall result arrays
num_algorithms = size(algorithms,1);
execution_times = zeros(num_algorithms,iterations);
counter = 0;
    
for i=1:iterations
        
    % generate experiment        
    [v1,v2,t,R] = create2D2DExperiment(pt_number,cam_number,noise,outlier_fraction);
    [t_perturbed,R_perturbed] = perturb(t,R,0.01);
    T_perturbed = [R_perturbed,t_perturbed];

    for a=1:num_algorithms
        tic
        Out = opengv(algorithms{a},indices{a},v1,v2,T_perturbed);
        execution_times(a,i) = toc/20.0;
    end

    counter = counter + 1;
    if counter == 100
        counter = 0;
        display(['Iteration ' num2str(i) ' of ' num2str(iterations)]);
    end
end

%% Plot the results
bins =[0.5e-6:1e-6:0.6e-4];
hist(execution_times', bins)
legend(names,'Location','NorthWest')
xlabel('execution times [s]')
grid on

mean(execution_times')
