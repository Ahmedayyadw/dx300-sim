% Stage 4 run - boom-up from th_min with the hydraulic model + static stall test.
% Needs dx300_boom_hyd.slx (run build_stage4.m first).
here = fileparts(mfilename('fullpath'));
run(fullfile(here,'dx300_params_lab08a.m'));
K = load(fullfile(here,'stage1_kin.mat'));            % th, L, r, tau_up
figdir = fullfile(here,'figs'); if ~exist(figdir,'dir'), mkdir(figdir); end
m = 'dx300_boom_hyd';  p_atm = 101325;                % Pa, standard atmosphere
t_spool = 1.0;                                        % must match spool_t in the model InitFcn
load_system(fullfile(here,[m '.slx']));

%% ---- 1. Full boom-up ------------------------------------------------------
set_param(m,'StopTime','9');
out = sim(m);
lg  = out.logsout;
g   = @(n) lg.get(n).Values;
q = g('q'); w = g('w'); pc = g('p_cap'); pp = g('p_pump'); pr = g('p_rod'); F = g('F'); tau = g('tau');
t = q.Time;
pc_g = pc.Data - p_atm;  pp_g = interp1(pp.Time,pp.Data,t) - p_atm;  pr_g = interp1(pr.Time,pr.Data,t) - p_atm;
pc_g = interp1(pc.Time,pc_g,t);

tol = deg2rad(0.2);                                   % "reached" = within 0.2 deg of th_max
i1 = find(q.Data >= P.th_max - tol, 1);
if isempty(i1), t_reach = NaN; else, t_reach = t(i1); end
t_up = t_reach - t_spool;
% pressure stats over the stroke only: spool opening .. first arrival at th_max (end-stop contact after that
% is a hard-stop impact artefact; the model has no cushioning) - impact peak reported separately
mov = t > t_spool & t <= t_reach;
imp = t > t_reach;
relief_open_motion = max(pp_g(mov)) >= P.p_relief - 0.5e5;   % within 0.5 bar of set point = valve cracking
fprintf('--- boom-up ---\n');
fprintf('theta(0) = %.2f deg (th_min %.2f), theta(end) = %.2f deg (th_max %.2f)\n', ...
    rad2deg(q.Data(1)), rad2deg(P.th_min), rad2deg(q.Data(end)), rad2deg(P.th_max));
fprintf('time to th_max after spool opens: %.2f s  (flow-only estimate %.2f s)\n', t_up, ...
    P.n_cyl*pi/4*P.bore^2*P.stroke/P.Q_pump);
fprintf('max cap pressure during stroke = %.1f bar (gauge); max pump-line pressure = %.1f bar\n', max(pc_g(mov))/1e5, max(pp_g(mov))/1e5);
fprintf('end-stop impact (after th_max reached): max cap pressure = %.1f bar, max pump line = %.1f bar\n', max(pc_g(imp))/1e5, max(pp_g(imp))/1e5);
fprintf('relief set %.0f bar; relief opened during stroke: %s; before spool opens (pump dead-headed, expected): %s; after th_max: %s\n', ...
    P.p_relief/1e5, yesno(relief_open_motion), yesno(max(pp_g(t<t_spool)) >= P.p_relief-0.5e5), yesno(max(pp_g(imp)) >= P.p_relief-0.5e5));
fprintf('max rod-side pressure = %.1f bar, max torque during stroke = %.0f kN m\n', max(pr_g(mov))/1e5, max(tau.Data(tau.Time>t_spool & tau.Time<=t_reach))/1e3);
fprintf('peak w = %.1f deg/s\n', rad2deg(max(w.Data(w.Time<=t_reach))));

figure('Position',[100 100 800 700]);
subplot(3,1,1); plot(t,rad2deg(q.Data),'LineWidth',1.5); grid on; yline(rad2deg([P.th_min P.th_max]),'--r');
ylabel('\theta [deg]'); title(sprintf('Boom-up from \\theta_{min}: %.2f s after spool opens', t_up));
subplot(3,1,2); plot(t,pc_g/1e5,t,pp_g/1e5,t,pr_g/1e5,'LineWidth',1.3); grid on; yline(P.p_relief/1e5,'--r');
ylabel('p [bar gauge]'); legend('cap','pump line','rod side','relief','Location','southeast');
subplot(3,1,3); kk = tau.Time <= t_reach; plot(tau.Time(kk),tau.Data(kk)/1e3,'LineWidth',1.3); grid on; ylabel('\tau [kN m]'); xlabel('t [s]');
for sp = 1:3, subplot(3,1,sp); xlim([0 t_reach+0.5]); end   % impact spike after th_max omitted
exportgraphics(gcf, fullfile(figdir,'stage4_boom_up.png'),'Resolution',200);


%% ---- 1b. Sensitivity: spool opened over 0.3 s instead of a step -----------
% (design check only; the spec is a step). Rate limiter after the Step in a temp copy.
tmp2 = 'stage4_ramp_tmp';
if bdIsLoaded(tmp2), close_system(tmp2,0); end
copyfile(fullfile(here,[m '.slx']), fullfile(here,[tmp2 '.slx']), 'f'); fileattrib(fullfile(here,[tmp2 '.slx']),'+w');
load_system(fullfile(here,[tmp2 '.slx']));
cleanup2 = onCleanup(@() stall_cleanup(tmp2, here));
ph = get_param([tmp2 '/Spool_SPS'],'PortHandles'); delete_line(get_param(ph.Inport(1),'Line'));
rl = add_block('simulink/Discontinuities/Rate Limiter',[tmp2 '/spool_rl'],'RisingSlewLimit','5e-3/0.3','FallingSlewLimit','-5e-3/0.3');   % 5e-3 = dS_max (ESTIMATED)
add_line(tmp2, get_param([tmp2 '/spool_cmd'],'PortHandles').Outport(1), get_param(rl,'PortHandles').Inport(1));
add_line(tmp2, get_param(rl,'PortHandles').Outport(1), ph.Inport(1));
set_param(tmp2,'StopTime','9');
o2 = sim(tmp2); l2 = o2.logsout;
q2 = l2.get('q').Values; pc2 = l2.get('p_cap').Values; pp2 = l2.get('p_pump').Values;
j2 = find(q2.Data >= P.th_max - tol, 1); tr2 = q2.Time(j2);
mv2 = pc2.Time > t_spool & pc2.Time <= tr2;
fprintf('--- sensitivity: spool ramp 0.3 s ---\n');
fprintf('time to th_max after spool opens: %.2f s; cap pressure during stroke: min %.1f, max %.1f bar\n', ...
    tr2 - t_spool, (min(pc2.Data(mv2))-p_atm)/1e5, (max(pc2.Data(mv2))-p_atm)/1e5);
fprintf('(step case: cap pressure during stroke min %.1f, max %.1f bar)\n', min(pc_g(mov))/1e5, max(pc_g(mov))/1e5);

%% ---- 2. Static stall test: tau_max(theta) at relief pressure --------------
% Temp copy: rod locked (v = 0) at x_i = L(theta_i)-L(th_min); torque path cut. Measure F.
tmp = 'stage4_stall_tmp';
if bdIsLoaded(tmp), close_system(tmp,0); end
copyfile(fullfile(here,[m '.slx']), fullfile(here,[tmp '.slx']), 'f'); fileattrib(fullfile(here,[tmp '.slx']),'+w');
load_system(fullfile(here,[tmp '.slx']));
cleanup = onCleanup(@() stall_cleanup(tmp, here));
for blk = {'v_SPS','tau_SPS'}
    ph = get_param([tmp '/' blk{1}],'PortHandles'); ln = get_param(ph.Inport(1),'Line'); delete_line(ln);
    c0 = add_block('simulink/Sources/Constant',[tmp '/zero_' blk{1}],'Value','0');
    add_line(tmp, get_param(c0,'PortHandles').Outport(1), ph.Inport(1));
end
set_param(tmp,'StopTime','4','UnconnectedOutputMsg','none');
idx = round(linspace(0.08,0.92,9)*numel(K.th));
Fst = zeros(size(idx)); pst = Fst;
for k = 1:numel(idx)
    xi = K.L(idx(k)) - K.L(1);
    set_param([tmp '/Cyl_eq'],'x0',sprintf('%.6f',xi));
    o = sim(tmp);
    Ft = o.logsout.get('F').Values;  pt = o.logsout.get('p_cap').Values;
    s = Ft.Time > 3;                                   % settled window 3..4 s
    Fst(k) = abs(mean(Ft.Data(s)));  pst(k) = mean(pt.Data(s)) - p_atm;
end
tau_st = Fst .* K.r(idx);
tau_ref = K.tau_up(idx);
err_pct = (tau_st - tau_ref)./tau_ref*100;
fprintf('--- static stall at %.0f bar ---\n', P.p_relief/1e5);
fprintf(' theta[deg]  p_cap[bar]  tau_model[kNm]  tau_up[kNm]  err[%%]\n');
fprintf(' %9.1f  %10.1f  %14.1f  %11.1f  %6.2f\n', [rad2deg(K.th(idx)); pst/1e5; tau_st/1e3; tau_ref/1e3; err_pct]);
fprintf('max |err| = %.2f %%\n', max(abs(err_pct)));
figure; plot(rad2deg(K.th),K.tau_up/1e3,'LineWidth',1.5); hold on
plot(rad2deg(K.th(idx)),tau_st/1e3,'o','MarkerSize',7,'LineWidth',1.5); grid on
xlabel('\theta [deg]'); ylabel('\tau_{max} [kN m]'); legend('stage1 \tau_{up}','Simscape stall','Location','best');
title(sprintf('Stall torque at relief: max error %.2f %%', max(abs(err_pct))));
exportgraphics(gcf, fullfile(figdir,'stage4_stall_torque.png'),'Resolution',200);

function s = yesno(b), if b, s = 'yes'; else, s = 'no'; end, end
function stall_cleanup(tmp, here)
    if bdIsLoaded(tmp), close_system(tmp,0); end
    f = fullfile(here,[tmp '.slx']); if exist(f,'file'), delete(f); end   % own temp file, inside lab08a
end
