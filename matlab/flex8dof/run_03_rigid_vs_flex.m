%% Option 3 - rigid vs flexible: same input, compare bucket tip height
% Python check: flexible tip overshoots/undershoots rigid by up to ~10 cm
clear; clc; close all;
p = dx300_params(); opt = odeset('RelTol',1e-7,'AbsTol',1e-9); tt = linspace(0,6,1201);
[~,xf] = ode15s(@(t,x) dyn_flex(t,x,p),  tt, [p.q0; zeros(4,1); p.th0; zeros(4,1)], opt);
[~,xr] = ode15s(@(t,x) dyn_rigid(t,x,p), tt, [p.q0; zeros(4,1)], opt);
zf = zeros(size(tt)); zr = zf;
for n = 1:numel(tt)
    [~,zf(n)] = tipz(xf(n,1:4)',p);  [~,zr(n)] = tipz(xr(n,1:4)',p);
end
subplot(2,1,1); plot(tt,zr,tt,zf,'--'); grid on; legend('rigid','flexible');
ylabel('bucket tip height [m]');
subplot(2,1,2); plot(tt,100*(zf-zr)); grid on; ylabel('flex - rigid [cm]'); xlabel('t [s]');
fprintf('Max tip difference = %.1f cm\n', 100*max(abs(zf-zr)));

function [r,z] = tipz(q,p)
ph = cumsum(q(2:4)); r = p.r0 + sum(p.L.*cos(ph')); z = p.z0 + sum(p.L.*sin(ph'));
end
