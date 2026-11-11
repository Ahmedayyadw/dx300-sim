% Lab 09b2 run - Simscape reference for the Python hydraulic energy model of Lab 09b.
% Runs boom-up (as lab08a stage4_run) and boom-down on the lab09b2 COPY of the hydraulic model,
% computes the Simscape energy split and exports the logs for rl/09b2_energy_check.py.
% Usage (from anywhere):  matlab -batch "run('matlab/lab09b2/lab09b2_run.m')"
% Nothing in matlab/lab08a/ is written.
%
% Value status: pump flow, relief pressure, bore, rod, stroke OFFICIAL (dx300_params_lab08a.m);
% valve area, spool stroke, oil data, dead volumes, pin geometry ESTIMATED (lab08a InitFcn);
% spool step time 1 s, 0.2 s pump ramp, 0.2 deg arrival tolerance: design choices of lab08a.
%
% Variants (base-workspace variables, see lab09b2_build.m; defaults = first run, undamped, step, full spool):
%   matlab -batch "src_model_09b2='dx300_boom_hyd_damped'; tag_09b2='_damped'; spool_frac_down=0.85; run('matlab/lab09b2/lab09b2_run.m')"
here09 = fileparts(mfilename('fullpath')); cd(here09);        % slprj / slxc cache stays in lab09b2
run(fullfile(here09,'lab09b2_build.m'));                      % sets m, tag_09b2, spool_ramp_09b2
expdir = fullfile(here09,'export'); if ~exist(expdir,'dir'), mkdir(expdir); end
load_system(fullfile(here09,[m '.slx']));
if ~exist('stop_time_09b2','var'), stop_time_09b2 = 9; end   % s; raise for slow (metered / ramped) moves
set_param(m,'StopTime',num2str(stop_time_09b2));
t_spool = 1.0;                                % must match spool_t in the model InitFcn
tol = deg2rad(0.2);                           % "reached" = within 0.2 deg of the end stop (as stage4_run)
dt_exp = 1e-3;                                % export grid [s]
if ~exist('spool_frac_down','var'), spool_frac_down = 1; end   % spool opening of the down move (1 = full, as up)

moves = struct('name',{'up','down'}, 'dir',{1,-1}, 'frac',{1, spool_frac_down});
rows = {};
for k = 1:numel(moves)
    move_09b2 = moves(k).dir; spool_frac_09b2 = moves(k).frac; %#ok<NASGU>   read by the model InitFcn (base workspace)
    tic; out = sim(m); wall = toc;
    R = lab09b2_energy_split(out, moves(k), t_spool, tol, evalin('base','P'), evalin('base','A_cap'), evalin('base','A_ann'));
    fprintf('=== %s%s (spool %+.0f %% of dS_max, ramp %g s), sim wall time %.1f s ===\n', moves(k).name, tag_09b2, 100*moves(k).dir*moves(k).frac, spool_ramp_09b2, wall);
    disp(struct2table(R.sum));
    T = lab09b2_export_table(R, dt_exp);
    writetable(T, fullfile(expdir, sprintf('simscape_%s%s.csv', moves(k).name, tag_09b2)));
    fprintf('wrote export/simscape_%s%s.csv  (%d rows, %.3f s grid)\n', moves(k).name, tag_09b2, height(T), dt_exp);
    rows{end+1} = R.sum; %#ok<SAGROW>
end
S = struct2table([rows{:}]);
writetable(S, fullfile(expdir,['simscape_energy' tag_09b2 '.csv']));
fprintf('wrote export/simscape_energy%s.csv\n', tag_09b2);
close_system(m,0);
