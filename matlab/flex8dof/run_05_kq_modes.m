%% Option 5a - pose-dependent boom stiffness k(q2) from the real DX300 pins
% Pins (same as the 3D model, Design A):  A boom foot, B cylinder base,
% C cylinder rod pin on boom, D boom-arm pin.
%   L(q2)   = |C(q2) - B|                     cylinder length
%   x       = L - Lmin                        extension (head side fills)
%   V_A     = A_A*x + Vh,  V_B = A_B*(stroke - x) + Vh
%   k(q2)   = beta*(A_A^2/V_A + A_B^2/V_B) * (dL/dq2)^2
% Python check: boom range ~ -24..86 deg, moment arm ~0.84 m at the 3D pose,
% k from ~3.4e8 (boom down) to ~6e6 N m/rad (boom up),
% boom mode from ~5.1 Hz (down) to ~2 Hz (up).
clear; clc; close all;
p = dx300_params();

A = [0 0.7]; B = [0.571 0.005]; C = [1.8461 2.7558]; D = [5.7103 3.2293];
qref = atan2(D(2)-A(2), D(1)-A(1));          % boom angle of A->D in this pose
rC   = C - A;
Lcyl = @(q) norm(A + ([cos(q-qref) -sin(q-qref); sin(q-qref) cos(q-qref)]*rC')' - B);

stroke = 1.45;
Lmin   = Lcyl(qref) - stroke/2;              % (PH) assumes 3D pose = mid-stroke; check retracted length in shop manual
AA = 2*pi/4*0.140^2;  AB = AA - 2*pi/4*0.095^2;   % two boom cylinders
e  = 1e-6;

% boom angles the cylinder can actually reach
qs = deg2rad(-70:0.1:90);  x = arrayfun(@(q) Lcyl(q)-Lmin, qs);
qs = qs(x >= 0 & x <= stroke);
fprintf('Boom range from cylinder stroke: %.1f to %.1f deg\n', rad2deg(qs(1)), rad2deg(qs(end)));

qq = linspace(qs(1), qs(end), 40);
[kq, arm, VA, VB] = deal(zeros(size(qq)));  F = zeros(numel(qq),4);
for n = 1:numel(qq)
    q2 = qq(n);  xn = Lcyl(q2) - Lmin;
    arm(n) = (Lcyl(q2+e) - Lcyl(q2-e))/(2*e);
    VA(n)  = AA*xn + p.Vh;  VB(n) = AB*(stroke-xn) + p.Vh;
    kq(n)  = p.beta*(AA^2/VA(n) + AB^2/VB(n)) * arm(n)^2;

    K = p.K;  K(2,2) = kq(n);                        % boom spring at this pose
    q = [0 q2 deg2rad(-80) deg2rad(-60)]';
    dG = zeros(4);
    for j = 1:4, d = zeros(4,1); d(j) = e; dG(:,j) = (Gvec(q+d,p)-Gvec(q-d,p))/(2*e); end
    F(n,:) = sort(sqrt(abs(eig(K + dG, Mmat(q,p)))))'/(2*pi);
end

b = rad2deg(qq);
figure('Name','Pose-dependent boom stiffness');
subplot(2,2,1); plot(b,arm); grid on; ylabel('moment arm dL/dq [m]');
subplot(2,2,2); plot(b,1e3*VA,b,1e3*VB); grid on; ylabel('oil volume [L]'); legend('V_A head','V_B rod');
subplot(2,2,3); semilogy(b,kq); grid on; ylabel('k_{boom} [N m/rad]'); xlabel('boom angle [deg]');
subplot(2,2,4); plot(b,F,'o-','MarkerSize',3); grid on; ylim([0 30]);
ylabel('natural freq [Hz]'); xlabel('boom angle [deg]'); legend('mode 1','mode 2','mode 3','mode 4');
