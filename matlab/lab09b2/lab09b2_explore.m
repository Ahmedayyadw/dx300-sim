% Lab 09b2 explore - which spool ramp and which boom-down opening give a clean Simscape reference
% on the lab08a damped model (dx300_boom_hyd_damped.slx: cylinder friction 1.9 kN + 2.5 kN s/m, ESTIMATED).
% Runs on its own copy (dx300_boom_hyd_09b2_explore.slx). Nothing in matlab/lab08a/ is written.
% Usage:  matlab -batch "run('matlab/lab09b2/lab09b2_explore.m')"
% Spool ramp and opening fraction are design choices (ASSUMED), not DX300 data.
here09 = fileparts(mfilename('fullpath')); cd(here09);
src_model_09b2 = 'dx300_boom_hyd_damped'; tag_09b2 = '_explore'; spool_ramp_09b2 = 1e-3;   % limiter present, 1 ms = step
run(fullfile(here09,'lab09b2_build.m'));
load_system(fullfile(here09,[m '.slx']));
set_param(m,'StopTime','12');
t_spool = 1.0; tol = deg2rad(0.2);
%        dir  frac  ramp[s]
cases = [  1  1.00  1e-3;   1  1.00  1.0;   1  1.00  2.0;   1  1.00  4.0; ...
          -1  1.00  1e-3;  -1  0.90  1e-3;  -1  0.85  1e-3;  -1  0.80  1e-3;  -1  0.75  1e-3;  -1  0.70  1e-3; ...
          -1  0.85  1.0;   -1  0.80  1.0;   -1  0.75  1.0];
fprintf('move  frac  ramp_s  area_mm2  t_move_s  p_cap min/max bar  p_rod min/max bar  cav_cap  cav_rod  osc_amp_bar  relief_frac  Qcyl_peak_Lmin  E_pump  E_relief  E_valve  E_mech_signed  W_pos kJ\n');
for k = 1:size(cases,1)
    move_09b2 = cases(k,1); spool_frac_09b2 = cases(k,2); spool_ramp_09b2 = cases(k,3); %#ok<NASGU>
    mv = struct('name', 'up', 'dir', cases(k,1), 'frac', cases(k,2)); if mv.dir < 0, mv.name = 'down'; end
    try
        out = sim(m); nlog = numel(out.simlog.p_pump.Pg.series.time);
        R = lab09b2_energy_split(out, mv, t_spool, tol, evalin('base','P'), evalin('base','A_cap'), evalin('base','A_ann'));
    catch err
        fprintf('%-4s  %.2f  %6.3f  FAILED: %s\n', mv.name, mv.frac, cases(k,3), strrep(err.message, newline, ' ')); continue
    end
    s = R.sum; w = R.t >= t_spool + 0.5 & R.t <= s.t_reach_s - 0.2;
    tu = (R.t(find(w,1)):1e-3:R.t(find(w,1,'last')))'; [tt, iu] = unique(R.t); pu = interp1(tt, R.p_cap(iu), tu);
    pd = pu - movmean(pu, 1000); amp = (max(pd) - min(pd))/2/1e5;          % as stage5 osc_metrics, without the edge trim
    fprintf('%-4s  %.2f  %6.3f  %8.1f  %8.2f  %7.1f / %5.1f    %7.2f / %5.1f    %7.3f  %7.3f  %11.1f  %11.3f  %14.0f  %6.1f  %8.1f  %7.1f  %13.1f  %8.1f\n', ...
        mv.name, mv.frac, cases(k,3), max(mv.frac-0.5,0)*3e-4*1e6, s.t_move_s, s.p_cap_min_bar, s.p_cap_max_bar, s.p_rod_min_bar, max(R.p_rod(R.t>=t_spool & R.t<=s.t_reach_s))/1e5, ...
        s.cav_cap_frac, s.cav_rod_frac, amp, s.relief_open_frac, s.Q_cyl_peak_Lmin, s.E_pump_kJ, s.E_relief_kJ, s.loss_supply_valve_kJ, s.E_mech_kJ, s.W_pos_kJ); fprintf('      log points %d, log t0 %.3f s\n', nlog, R.t(1));
end
close_system(m,0);
