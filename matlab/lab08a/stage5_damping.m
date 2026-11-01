% Stage 5 - damping of the hydraulic resonance, sensitivity, parameter export for rl/boom_env.py
% Run:  matlab -batch "cd('<lab08a>'); stage5_damping"      (takes several minutes: 6 boom-up sims)
% Everything not OFFICIAL is ESTIMATED and labelled as such.
build_stage4;                                          % rebuilds dx300_boom_hyd.slx from dx300_boom_mb.slx (clobbers workspace)
close_system('dx300_boom_hyd',0);
clearvars -except here;
here = fileparts(mfilename('fullpath'));
run(fullfile(here,'dx300_params_lab08a.m'));           % struct P
K  = load(fullfile(here,'stage1_kin.mat'));            % th, L, r, tau_up, tau_down
S2 = load(fullfile(here,'stage2_mass.mat'));           % Lc0, J_A0
figdir = fullfile(here,'figs');  expdir = fullfile(here,'export');
if ~exist(figdir,'dir'), mkdir(figdir); end
if ~exist(expdir,'dir'), mkdir(expdir); end
p_atm = 101325;                                        % Pa, standard atmosphere
t_spool = 1.0;                                         % s, spool step time (build_stage4 InitFcn spool_t)
tol = deg2rad(0.2);                                    % "reached th_max" tolerance

%% ---- constants from Stage 3 (re-derived, checked against the values quoted in the task) -----
La = 3.1; phi_rel = deg2rad(-90);                      % arm length OFFICIAL (M p.62); arm perpendicular to boom = ESTIMATED pose
m_Dpt = 340; m_armpt = 1050+220; m_tippt = 1120+1.27*1800;   % kg; OFFICIAL masses (M p.65-66, S5 p.21), payload density 1800 kg/m^3 ESTIMATED
J_brick = P.m_boom*(P.L_boom^2+0.5^2)/12 + P.m_boom*S2.Lc0^2;          % ESTIMATED (brick + Lc0 = L/2), as stage3_validate.m
pts = [P.L_boom 0; P.L_boom+0.5*La*cos(phi_rel) 0.5*La*sin(phi_rel); P.L_boom+La*cos(phi_rel) La*sin(phi_rel)];
J_front = sum([m_Dpt m_armpt m_tippt]'.*sum(pts.^2,2));                 % ESTIMATED
J_total = J_brick + J_front;
fprintf('J_brick = %.0f, J_front = %.0f, J_total = %.0f kg m^2 (task: 30339 / 231407 / 261746)\n', J_brick, J_front, J_total);
assert(abs(J_total-261746) < 5, 'J_total differs from the Stage-3 value');
J_range = [128479 420986];                             % ESTIMATED, Stage 2 (J_A over arm poses) - copied from task text
th_mid = (P.th_min+P.th_max)/2;
r_mid  = interp1(K.th, K.r, th_mid);                   % m, ESTIMATED (pins)
m_eq   = J_total/r_mid^2;                              % kg, ESTIMATED equivalent rod mass

%% ---- case runner -----------------------------------------------------------
cases = struct('name',{},'label',{},'friction',{},'preload',{},'visc',{});
cases(1) = struct('name','undamped','label','undamped','friction',false,'preload',0,'visc',0);
% friction only: ESTIMATED, Li 2020 Table 3 p.13 (600 N, 800 N s/m, D90, 4 t) x (140/90) x 2 cylinders
Fc = 600*(140/90)*2;  cv_fr = 800*(140/90)*2;           % -> ~1.87 kN, ~2.49 kN s/m  (task rounds to 1.9 / 2.5)
Fc = 1.9e3;  cv_fr = 2.5e3;                             % ESTIMATED, rounded as specified
cases(2) = struct('name','friction','label','friction only','friction',true,'preload',Fc,'visc',cv_fr);

% damped model copy requested by the task: dx300_boom_hyd_damped.slx (friction-only)
dm = 'dx300_boom_hyd_damped';
if bdIsLoaded(dm), close_system(dm,0); end
copyfile(fullfile(here,'dx300_boom_hyd.slx'), fullfile(here,[dm '.slx']), 'f'); fileattrib(fullfile(here,[dm '.slx']),'+w');
load_system(fullfile(here,[dm '.slx']));
set_param([dm '/Cyl_eq'], 'friction_spec','true', 'preload_force','1.9e3', 'preload_force_unit','N', ...   % Coulomb force 1.9 kN ESTIMATED
    'Coulomb_coeff','0', 'Coulomb_coeff_unit','N/Pa', ...                  % no pressure-dependent Coulomb part
    'viscous_coeff','2.5e3', 'viscous_coeff_unit','N*s/m', ...             % ESTIMATED
    'breakaway_Coulomb_ratio','1.0');                                      % ESTIMATED (no stiction peak)
save_system(dm); close_system(dm,0);

R = struct();
R.undamped = run_case(here, 'dx300_boom_hyd', cases(1), P, p_atm, t_spool, tol);
R.friction = run_case(here, dm,                 [],       P, p_atm, t_spool, tol);   % model already carries friction
fprintf('undamped: t_up=%.2f s, f_osc=%.3f Hz, amp=%.1f bar\n', R.undamped.t_up, R.undamped.f, R.undamped.amp/1e5);
fprintf('friction: t_up=%.2f s, f_osc=%.3f Hz, amp=%.1f bar\n', R.friction.t_up, R.friction.f, R.friction.amp/1e5);

%% ---- Sensitivity: viscous term for damping ratio zeta ----------------------
wn = 2*pi*R.undamped.f;                                % rad/s, measured from undamped run (dominant p_cap oscillation)
zetas = [0.05 0.1 0.2];
c_z = 2*zetas*m_eq*wn;                                  % N s/m on the rod, ESTIMATED
fprintf('m_eq = J_total/r_mid^2 = %.0f kg (r_mid = %.3f m), w_n = %.2f rad/s\n', m_eq, r_mid, wn);
for i = 1:3
    nm = sprintf('zeta%03d', round(zetas(i)*100));
    cs = struct('name',nm,'label',sprintf('\\zeta = %.2f',zetas(i)),'friction',true,'preload',0,'visc',c_z(i));
    R.(nm) = run_case(here, 'dx300_boom_hyd', cs, P, p_atm, t_spool, tol);
    fprintf('zeta=%.2f: c=%.3g N s/m, t_up=%.2f s, p_cap max/min = %.1f/%.1f bar, amp=%.1f bar, f=%.3f Hz\n', ...
        zetas(i), c_z(i), R.(nm).t_up, R.(nm).pmax/1e5, R.(nm).pmin/1e5, R.(nm).amp/1e5, R.(nm).f);
end

%% ---- Figure -----------------------------------------------------------------
names = {'undamped','friction','zeta005','zeta010','zeta020'};
labs  = {'undamped','friction only','\zeta = 0.05','\zeta = 0.10','\zeta = 0.20'};
col = [0.35 0.35 0.35; 0.85 0.33 0.10; 0.00 0.45 0.74; 0.47 0.67 0.19; 0.49 0.18 0.56];
figure('Position',[100 100 900 700]);
ax1 = subplot(2,1,1); hold on; grid on; ax2 = subplot(2,1,2); hold on; grid on;
for i = 1:numel(names)
    d = R.(names{i});
    plot(ax1, d.t, rad2deg(d.q), 'Color',col(i,:), 'LineWidth',1.4);
    plot(ax2, d.tp, (d.pc-p_atm)/1e5, 'Color',col(i,:), 'LineWidth',1.1);
end
yline(ax1, rad2deg([P.th_min P.th_max]), ':k'); yline(ax2, P.p_relief/1e5, ':k', 'relief');
ylabel(ax1,'\theta [deg]'); ylabel(ax2,'cap pressure [bar gauge]'); xlabel(ax2,'t [s]');
legend(ax1, labs, 'Location','northwest'); title(ax1,'Boom-up from \theta_{min}: damping comparison (all damping ESTIMATED)');
xlim(ax1,[0 12]); xlim(ax2,[0 12]);
exportgraphics(gcf, fullfile(figdir,'stage5_damping_compare.png'),'Resolution',200);

%% ---- JSON export for rl/boom_env.py ------------------------------------------
% Rotational viscous equivalents: b = c * r(theta)^2   [N m s/rad]  (virtual work: v = r*w, tau = F*r)
idx = round(linspace(1,numel(K.th),51));
b_fr_tab = cv_fr * K.r(idx).^2;                         % ESTIMATED (friction-only viscous term only; Coulomb part not representable)
b_z      = c_z * r_mid^2;                               % ESTIMATED, N m s/rad at theta_mid
fld = @(v,u,st,src) struct('value',v,'unit',u,'status',st,'source',src);
E = struct();
E.m_boom   = fld(P.m_boom,'kg','official','dx300_params_lab08a.m P.m_boom');
E.L_boom   = fld(P.L_boom,'m','official','dx300_params_lab08a.m P.L_boom (rl/ CLAUDE.md: 6.245 m)');
E.Lc       = fld(S2.Lc0,'m','estimated','stage2_mass.m Lc0 = L_boom/2 (CoM at mid-boom)');
E.J_boom_only    = fld(J_brick,'kg m^2','estimated','stage3_validate.m J_brick: brick about A, Lc0=L/2');
E.J_with_front   = fld(J_total,'kg m^2','estimated','stage3_validate.m J_brick+J_front, arm perpendicular to boom (phi_rel=-90 deg)');
E.J_range_vs_pose= fld(J_range,'kg m^2','estimated','Stage 2 (stage2_mass.m) J_A over arm poses [min max]');
E.front_masses   = fld(struct('m_D_point_kg',m_Dpt,'m_arm_point_kg',m_armpt,'m_tip_point_kg',m_tippt,'arm_length_m',La,'arm_angle_rel_boom_rad',phi_rel), ...
    'kg / m / rad','estimated','stage2_mass.m: masses OFFICIAL (M p.62,65,66; S5 p.21), lumping/pose/payload density 1800 kg/m^3 ESTIMATED');
E.th_min = fld(P.th_min,'rad','official','dx300_params_lab08a.m (per param_audit)');
E.th_max = fld(P.th_max,'rad','official','dx300_params_lab08a.m (per param_audit)');
E.tau_up   = fld(struct('theta_rad',K.th(idx),'tau_Nm',K.tau_up(idx)),  'rad / N m','estimated', ...
    'stage1_kin.mat: n_cyl*p_relief*A_cap*r(theta), 330 bar, no losses; bore/p OFFICIAL, moment arm r from ESTIMATED pins');
E.tau_down = fld(struct('theta_rad',K.th(idx),'tau_Nm',K.tau_down(idx)),'rad / N m','estimated', ...
    'stage1_kin.mat: n_cyl*p_relief*A_annulus*r(theta), 330 bar, no losses');
E.friction_cylinder_eq = fld(struct('coulomb_N',Fc,'viscous_Ns_per_m',cv_fr,'breakaway_ratio',1.0),'N / N s/m / -','estimated', ...
    'Li 2020 Table 3 p.13 (600 N, 800 N s/m, D90, 4 t) scaled by 140/90 and x2 cylinders; rod-side equivalent cylinder');
E.b_friction_rot_table = fld(struct('theta_rad',K.th(idx),'b_Nms_per_rad',b_fr_tab),'rad / N m s/rad','estimated', ...
    'c_visc*r(theta)^2 with c_visc = 2.5 kN s/m (ESTIMATED, Li 2020 scaled); Coulomb part (1.9 kN) not representable as b');
E.b_friction_rot_mean  = fld(mean(b_fr_tab),'N m s/rad','estimated','mean of b_friction_rot_table over theta');
E.b_zeta_candidates = fld(struct('zeta',zetas,'c_rod_Ns_per_m',c_z,'b_Nms_per_rad',b_z, ...
    'omega_n_rad_s',wn,'m_eq_kg',m_eq,'r_theta_mid_m',r_mid),'- / N s/m / N m s/rad','estimated', ...
    'c = 2*zeta*m_eq*w_n, m_eq=J_total/r(theta_mid)^2, w_n measured from undamped Simscape run; b = c*r(theta_mid)^2');
E.boom_up_time_s = fld(R.undamped.t_up,'s','estimated','stage5_damping.m undamped Simscape run (spool step at 1 s to th_max-0.2 deg); flow-only 5.42 s');
E.boom_up_time_friction_s = fld(R.friction.t_up,'s','estimated','stage5_damping.m friction-only run');
E.Q_pump   = fld(P.Q_pump,'m^3/s','official','dx300_params_lab08a.m (494 L/min)');
E.p_relief = fld(P.p_relief,'Pa','official','dx300_params_lab08a.m (330 bar)');
txt = jsonencode(E,'PrettyPrint',true);
fid = fopen(fullfile(expdir,'boom_params_lab08a.json'),'w'); fwrite(fid,txt); fclose(fid);
save(fullfile(here,'stage5_results.mat'),'R','c_z','zetas','wn','m_eq','r_mid');
fprintf('wrote %s\n', fullfile(expdir,'boom_params_lab08a.json'));

%% ---- local functions ----------------------------------------------------------
function d = run_case(here, srcmodel, cs, P, p_atm, t_spool, tol)
    tmp = 'stage5_tmp';
    if bdIsLoaded(tmp), close_system(tmp,0); end
    copyfile(fullfile(here,[srcmodel '.slx']), fullfile(here,[tmp '.slx']), 'f'); fileattrib(fullfile(here,[tmp '.slx']),'+w');
    load_system(fullfile(here,[tmp '.slx']));
    cl = onCleanup(@() cleanup_tmp(tmp, here));
    if ~isempty(cs) && cs.friction
        set_param([tmp '/Cyl_eq'], 'friction_spec','true', 'preload_force',sprintf('%.9g',cs.preload), 'preload_force_unit','N', ...
            'Coulomb_coeff','0', 'Coulomb_coeff_unit','N/Pa', 'viscous_coeff',sprintf('%.9g',cs.visc), 'viscous_coeff_unit','N*s/m', ...
            'breakaway_Coulomb_ratio','1.0');
    end
    set_param(tmp,'StopTime','12','UnconnectedOutputMsg','none');
    o = sim(tmp); lg = o.logsout;
    q = lg.get('q').Values; pc = lg.get('p_cap').Values;
    i1 = find(q.Data >= P.th_max - tol, 1);
    if isempty(i1), t_reach = NaN; t_end = q.Time(end); else, t_reach = q.Time(i1); t_end = t_reach; end
    d.t_up = t_reach - t_spool;  d.t = q.Time; d.q = q.Data; d.tp = pc.Time; d.pc = pc.Data; d.t_reach = t_reach;
    mv = pc.Time > t_spool & pc.Time <= t_end;
    d.pmax = max(pc.Data(mv)) - p_atm;  d.pmin = min(pc.Data(mv)) - p_atm;          % gauge, during stroke only
    [d.amp, d.f] = osc_metrics(pc.Time, pc.Data, t_spool+0.5, t_end-0.2);
end

function [amp, f] = osc_metrics(t, p, t0, t1)
    % oscillation = p_cap minus 1 s moving mean; amplitude = half peak-to-peak; f = dominant FFT peak above 0.5 Hz
    amp = NaN; f = NaN;
    if ~(t1 - t0 > 3), return, end
    tu = (t0:1e-3:t1)';  pu = interp1(t, p, tu);
    pd = pu - movmean(pu, 1000);  pd = pd(501:end-500);
    amp = (max(pd) - min(pd))/2;
    N = 2^nextpow2(4*numel(pd));  Y = abs(fft(pd.*hann(numel(pd)), N));  fa = (0:N-1)'/N*1000;
    k = fa > 0.5 & fa < 50;  [~, j] = max(Y(k));  fk = fa(k);  f = fk(j);
end

function cleanup_tmp(tmp, here)
    if bdIsLoaded(tmp), close_system(tmp,0); end
    for e = {'.slx','.slxc'}
        f = fullfile(here,[tmp e{1}]); if exist(f,'file'), delete(f); end   % own temp files inside lab08a
    end
end
