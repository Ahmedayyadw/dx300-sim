%% Option 1 - boom only, 2 DOF (link angle q + actuator angle th, one oil spring)
% Python check: boom settles at 35.0 deg, natural frequency ~7.1 Hz
clear; clc; close all;
p  = dx300_params();
m = p.m(1); lc = p.lc(1); Mb = m*lc^2 + p.Ic(1);      % boom alone about pin A
kb = p.k(2); Jb = p.J(2,2); Bb = p.B(2,2);
Gb = @(q) m*p.g*lc*cos(q);
q0 = deg2rad(30); t0 = Gb(q0); th0 = q0 + t0/kb;
Db = 2*0.05*sqrt(kb*Mb); Kp = 0.5*kb; Kd = 2*0.9*sqrt(Kp*(Jb+Mb));
f = @(t,x) [ x(2);
            (kb*(x(3)-x(1)) + Db*(x(4)-x(2)) - Gb(x(1)))/Mb;
             x(4);
            (t0 + Kp*(th0 + deg2rad(5)*(t>=1) - x(3)) - Kd*x(4) - Bb*x(4) ...
              - kb*(x(3)-x(1)) - Db*(x(4)-x(2)))/Jb ];
[t,x] = ode15s(f,[0 4],[q0 0 th0 0],odeset('RelTol',1e-8,'AbsTol',1e-10));
subplot(2,1,1); plot(t,rad2deg(x(:,1)),t,rad2deg(x(:,3)),'--'); grid on
ylabel('angle [deg]'); legend('q boom (link)','\theta (actuator)'); title('Boom 2-DOF');
subplot(2,1,2); plot(t,rad2deg(x(:,3)-x(:,1))); grid on
ylabel('spring twist [deg]'); xlabel('t [s]');
fprintf('Boom alone natural frequency = %.2f Hz\n', sqrt(kb/Mb)/(2*pi));
