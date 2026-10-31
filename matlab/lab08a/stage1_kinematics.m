% Stage 1 - boom kinematics: cylinder length L(theta), moment arm r(theta)
clear; run(fullfile(fileparts(mfilename('fullpath')),'dx300_params_lab08a.m'));

% C in boom-fixed polar coords (rotates with boom about A)
rC   = norm(P.C);                          % ESTIMATED (from pins)
phiC = atan2(P.C(2),P.C(1)) - P.theta_ref; % boom-fixed angle of C

th = linspace(P.th_min, P.th_max, 500);
Cx = rC*cos(phiC+th);  Cy = rC*sin(phiC+th);
dx = Cx - P.B(1);      dy = Cy - P.B(2);
L  = hypot(dx,dy);                          % cylinder pin-to-pin length [m]
r  = abs(P.B(1)*Cy - P.B(2)*Cx)./L;         % moment arm about A [m] (= dL/dtheta)

% Stroke check vs official cylinder
fprintf('L at th_min = %.3f m (closed  %.3f, OFFICIAL)\n', L(1),  P.L_closed);
fprintf('L at th_max = %.3f m (extended %.3f, OFFICIAL)\n', L(end), P.L_closed+P.stroke);
fprintf('r range     = %.3f .. %.3f m\n', min(r), max(r));

figure; 
subplot(2,1,1); plot(rad2deg(th),L,'LineWidth',1.5); grid on
yline([P.L_closed P.L_closed+P.stroke],'--r'); ylabel('L_{cyl} [m]')
title('DX300 boom kinematics (pins ESTIMATED)')
subplot(2,1,2); plot(rad2deg(th),r,'LineWidth',1.5); grid on
ylabel('moment arm r [m]'); xlabel('\theta [deg]')
save(fullfile(fileparts(mfilename('fullpath')),'stage1_kin.mat'),'th','L','r')

% --- Static torque capacity (no losses) and stroke time ---
A_cap = pi/4*P.bore^2;                       % OFFICIAL-derived [m^2]
A_ann = A_cap - pi/4*P.rod^2;                % OFFICIAL-derived [m^2]
tau_up   = P.n_cyl*P.p_relief*A_cap.*r;      % boom-up limit [N m]
tau_down = P.n_cyl*P.p_relief*A_ann.*r;      % boom-down limit [N m]

V_up = P.n_cyl*A_cap*P.stroke;               % oil to extend both cyl [m^3]
fprintf('tau_up   = %.0f .. %.0f kN m\n', min(tau_up)/1e3,  max(tau_up)/1e3);
fprintf('tau_down = %.0f .. %.0f kN m\n', min(tau_down)/1e3, max(tau_down)/1e3);
fprintf('V_up = %.1f L, t_up @494 L/min = %.2f s, Q for 3.6 s = %.0f L/min\n', ...
    V_up*1e3, V_up/P.Q_pump, V_up/3.6*60000);

figure; plot(rad2deg(th),tau_up/1e3, rad2deg(th),tau_down/1e3,'LineWidth',1.5); grid on
legend('boom up (cap side)','boom down (rod side)'); xlabel('\theta [deg]')
ylabel('\tau_{max} [kN m]'); title('Static torque limit at 330 bar (no losses)')
save(fullfile(fileparts(mfilename('fullpath')),'stage1_kin.mat'),'th','L','r','tau_up','tau_down')

% --- Export figures ---
outdir = fullfile(fileparts(mfilename('fullpath')),'figs');
if ~exist(outdir,'dir'), mkdir(outdir); end
exportgraphics(figure(1),fullfile(outdir,'stage1_L_r_vs_theta.png'),'Resolution',200)
exportgraphics(gcf,      fullfile(outdir,'stage1_tau_max_vs_theta.png'),'Resolution',200)

