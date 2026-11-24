%% Option 2 - full 8-DOF flexible-joint model (swing, boom, arm, bucket)
% Python check: equilibrium |xdot| ~1e-11, energy drift ~1e-11 (undamped),
% reaches swing 10 deg / boom 35 deg, boom spring twist up to ~0.9 deg
clear; clc; close all;
p  = dx300_params();
x0 = [p.q0; zeros(4,1); p.th0; zeros(4,1)];
[t,x] = ode15s(@(t,x) dyn_flex(t,x,p), [0 6], x0, odeset('RelTol',1e-6,'AbsTol',1e-8));
q = x(:,1:4); th = x(:,9:12); names = {'Swing','Boom','Arm','Bucket'};
for i = 1:4
    subplot(4,2,2*i-1); plot(t,rad2deg(q(:,i)),t,rad2deg(th(:,i)),'--'); grid on
    ylabel([names{i} ' [deg]']); if i==1, legend('q (link)','\theta (actuator)'); end
    subplot(4,2,2*i); plot(t,rad2deg(th(:,i)-q(:,i)-(p.th0(i)-p.q0(i)))); grid on
    ylabel('\Delta spring [deg]');
end
xlabel('t [s]');
fprintf('k = [%s] N m/rad\n', num2str(p.k,'%.2e  '));
