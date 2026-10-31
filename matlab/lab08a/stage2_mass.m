% Stage 2 - boom mass properties: Lc, J about A, gravity torque
clear; here = fileparts(mfilename('fullpath'));
run(fullfile(here,'dx300_params_lab08a.m')); load(fullfile(here,'stage1_kin.mat'))
g = 9.81;  m = P.m_boom;  Lb = P.L_boom;          % OFFICIAL

% CoM model (ESTIMATED): fraction s along A->D, offset h perpendicular (up)
s_list = [0.40 0.45 0.50];   h_list = [0 0.3];    % ESTIMATED sweep
J_cm   = m*Lb^2/12;                               % ESTIMATED (slender rod)
m_cyl_rot = P.n_cyl*P.m_cyl/2;                    % ESTIMATED: half cyl mass at C
rC = norm(P.C);

fprintf('  s     h[m]   Lc[m]   J_A[kg m^2]  tau_g,max[kN m]\n')
for s = s_list
    for h = h_list
        Lc  = hypot(s*Lb, h);
        J_A = J_cm + m*Lc^2 + m_cyl_rot*rC^2;
        fprintf('%5.2f  %4.1f  %6.3f   %9.0f     %6.1f\n', s, h, Lc, J_A, (m*Lc+m_cyl_rot*rC)*g/1e3)
    end
end

% Baseline (s=0.5, h=0)
Lc0   = 0.5*Lb;
phiCm = atan2(P.C(2),P.C(1)) - P.theta_ref;
tau_g = m*g*Lc0*cos(th) + m_cyl_rot*g*rC*cos(th+phiCm);
J_A0  = J_cm + m*Lc0^2 + m_cyl_rot*rC^2;
save(fullfile(here,'stage2_mass.mat'),'Lc0','tau_g','J_A0')

% --- Real front attachment (3.1 m arm) at 3 arm poses ---
La     = 3.10;              % arm length, OFFICIAL M p.62 (pin-to-pin ESTIMATED = this)
m_arm  = 1050;              % OFFICIAL M p.66 (3.1 m arm)
m_acyl = 340;               % arm cylinder, OFFICIAL M p.66 -> lumped at D (ESTIMATED)
m_bcyl = 220;               % bucket cylinder, OFFICIAL M p.66 -> at arm mid (ESTIMATED)
m_bkt  = 1120;              % bucket 1.27 m3, OFFICIAL S5 p.21 (model unverified)
m_pay  = 1.27*1800;         % 1.27 m3 OFFICIAL M p.65 x 1800 kg/m3 ESTIMATED = 2286 kg
psi_list = deg2rad([-90 -45 0]);   % absolute arm angle from horizontal, ESTIMATED poses

figure; hold on; grid on
plot(rad2deg(th),tau_up/1e3,'k','LineWidth',2)
fprintf('\n psi[deg]  tau_g,max[kN m]  min margin[kN m]  J_A range[kg m^2]\n')
for psi = psi_list
    xD  = Lb*cos(th);           yD  = Lb*sin(th);
    xAm = xD + 0.5*La*cos(psi); yAm = yD + 0.5*La*sin(psi);
    xT  = xD + La*cos(psi);     yT  = yD + La*sin(psi);
    tau_f   = g*( m_acyl*xD + (m_arm+m_bcyl)*xAm + (m_bkt+m_pay)*xT );
    tau_tot = tau_g + tau_f;
    J_tot = J_A0 + m_acyl*Lb^2 + m_arm*La^2/12 ...
        + (m_arm+m_bcyl)*(xAm.^2+yAm.^2) + (m_bkt+m_pay)*(xT.^2+yT.^2);
    plot(rad2deg(th),tau_tot/1e3,'LineWidth',1.3)
    fprintf('%7.0f   %10.1f       %10.1f       %6.0f-%6.0f\n', rad2deg(psi), ...
        max(tau_tot)/1e3, min(tau_up-tau_tot)/1e3, min(J_tot), max(J_tot))
end
legend('\tau_{up} 330 bar','arm vertical (\psi=-90°)','\psi=-45°','arm horizontal (\psi=0°)')
xlabel('\theta [deg]'); ylabel('[kN m]'); title('Gravity torque, real 3.1 m arm + full bucket')
