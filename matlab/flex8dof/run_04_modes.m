%% Option 4 - natural frequencies vs boom angle (valves closed: th fixed)
% Linearize: M(q) dq'' + (K + dG/dq) dq = 0  ->  w^2 = eig(K + dG/dq, M)
% Python check: ~1.5, 3.0, 5.9, 25.3 Hz, almost flat because k is fixed at mid-stroke
clear; clc; close all;
p = dx300_params(); b = -10:10:60; F = zeros(numel(b),4);
for n = 1:numel(b)
    q = deg2rad([0 b(n) -80 -60])';
    dG = zeros(4); e = 1e-6;
    for j = 1:4, d = zeros(4,1); d(j) = e; dG(:,j) = (Gvec(q+d,p)-Gvec(q-d,p))/(2*e); end
    F(n,:) = sort(sqrt(abs(eig(p.K + dG, Mmat(q,p)))))'/(2*pi);
end
plot(b,F,'o-'); grid on; xlabel('boom angle q_2 [deg]'); ylabel('natural frequency [Hz]');
legend('mode 1','mode 2','mode 3','mode 4'); disp(array2table(F,'RowNames',string(b)));
