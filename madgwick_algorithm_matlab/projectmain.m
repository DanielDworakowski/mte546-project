%% Start of script

addpath('quaternion_library');      % include quaternion library
close all;                          % close all figures
clear;                              % clear all variables
clc;                                % clear the command terminal

%% Import and plot sensor data

data = csvread('~/Desktop/4B/MTE546/Project/v201-easy/imu0/data.csv',2);
dataGT=csvread('~/Desktop/4B/MTE546/Project/v201-easy/state_groundtruth_estimate0/data.csv',2);
dataVal=csvread('~/Desktop/4B/MTE546/Project/v201-easy/valenti/result.csv');

% find the common indexes
ix_start = find(data(:,1) == dataGT(1,1));
ix_end = find(data(:,1) == dataGT(end,1));
data = data(ix_start:ix_end, :);

time = data(:,1);
Accelerometer = data(:,5:7);
gyro_bias=[-0.002, 0.020933373, 0.081622879];
Gyroscope = data(:,2:4) - gyro_bias;

time_gt = dataGT(:,1);
quaternion_gt = dataGT(:,5:8);
Accelerometer_gt = dataGT(:,15:17);
Gyroscope_gt = dataGT(:,12:14);

quaternion_val = dataVal(2:end,:);

% figure('Name', 'Sensor Data');
% axis(1) = subplot(2,1,1);
% hold on;
% plot(time, Gyroscope(:,1), 'r');
% plot(time, Gyroscope(:,2), 'g');
% plot(time, Gyroscope(:,3), 'b');
% legend('X', 'Y', 'Z');
% xlabel('Time (s)');
% ylabel('Angular rate (deg/s)');
% title('Gyroscope');
% hold off;
% axis(2) = subplot(2,1,2);
% hold on;
% plot(time, Accelerometer(:,1), 'r');
% plot(time, Accelerometer(:,2), 'g');
% plot(time, Accelerometer(:,3), 'b');
% legend('X', 'Y', 'Z');
% xlabel('Time (s)');
% ylabel('Acceleration (g)');
% title('Accelerometer');
% hold off;

% axis(3) = subplot(3,1,3);
% hold on;
% plot(time, Magnetometer(:,1), 'r');
% plot(time, Magnetometer(:,2), 'g');
% plot(time, Magnetometer(:,3), 'b');
% legend('X', 'Y', 'Z');
% xlabel('Time (s)');
% ylabel('Flux (G)');
% title('Magnetometer');
% hold off;
%linkaxes(axis, 'x');

%% Process sensor data through algorithm

AHRS = MadgwickAHRS('SamplePeriod', 0.005,'Beta', 0.1);

quaternion = zeros(length(time), 4);
for t = 1:length(time)
    if t == 1
        timestep = 0.005;
    else
        timestep = (time(t)-time(t-1))/(1e+9);
    end
    AHRS.UpdateIMU(Gyroscope(t,:), Accelerometer(t,:), timestep);	% gyroscope units must be radians
    quaternion(t, :) = AHRS.Quaternion;
end

%% Plot algorithm output as Euler angles
% The first and third Euler angles in the sequence (phi and psi) become
% unreliable when the middle angles of the sequence (theta) approaches ±90
% degrees. This problem commonly referred to as Gimbal Lock.
% See: http://en.wikipedia.org/wiki/Gimbal_lock

euler = quatern2euler(quaternConj(quaternion));	% use conjugate for sensor frame relative to Earth and convert to degrees.
euler_gt = quatern2euler(quaternConj(quaternion_gt));
euler_val = quatern2euler(quaternConj(quaternion_val));

tseries = 1:length(time);

fontsize = 15;

figure('Name', 'Euler Angles \phi');
plot(tseries, euler(:,1), 'r', tseries, euler_gt(:,1), 'g', tseries, euler_val(:,1), 'b');
title('Roll', 'FontSize', fontsize);
grid on;
xlabel('Time Step', 'FontSize', fontsize);
ylabel('Angle (rad)', 'FontSize', fontsize);
legend({'Madgwick', 'Ground Truth','Valenti'}, 'FontSize', fontsize);

figure('Name', 'Euler Angles \theta');
plot(tseries, euler(:,2), 'r', tseries, euler_gt(:,2), 'g', tseries, euler_val(:,2), 'b');
title('Pitch', 'FontSize', fontsize);
grid on;
xlabel('Time Step', 'FontSize', fontsize);
ylabel('Angle (rad)', 'FontSize', fontsize);
legend({'Madgwick', 'Ground Truth','Valenti'}, 'FontSize', fontsize);

figure('Name', 'Euler Angles \psi');
plot(tseries, euler(:,3), 'r', tseries, euler_gt(:,3), 'g', tseries, euler_val(:,3), 'b');
title('Yaw', 'FontSize', fontsize);
grid on;
xlabel('Time Step', 'FontSize', fontsize);
ylabel('Angle (rad)', 'FontSize', fontsize);
legend({'Madgwick', 'Ground Truth','Valenti'}, 'FontSize', fontsize);

%% Error
error_madgwick = acos(2 .* (dot(quaternion', quaternion_gt')').^2 - 1);
error_valenti = acos(2 .* (dot(quaternion_val', quaternion_gt')').^2 - 1);

figure, plot(abs(error_madgwick), 'r');
hold on;
plot(abs(error_valenti), 'b');
legend({'Madgwick', 'Valenti'}, 'FontSize', fontsize);
title('Error',  'FontSize', fontsize);
xlabel('Time Step',  'FontSize', fontsize);
ylabel('Error (rad)',  'FontSize', fontsize);
grid on;

fprintf("Error in madgwick = %f\n", mean(abs(error_madgwick)));
fprintf("Error in Valenti = %f\n", mean(abs(error_valenti)));
