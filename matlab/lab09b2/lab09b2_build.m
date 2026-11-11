% Lab 09b2 build - copy of the lab08a hydraulic boom with extra logging and a selectable move.
% Idempotent: rebuilds matlab/lab09b2/dx300_boom_hyd_09b2.slx from a fresh copy of
% matlab/lab08a/dx300_boom_hyd.slx. Nothing in matlab/lab08a/ is written (read-only source).
%
% Changes on the copy only:
%   1. pump flow command (pump_sat output) logged as Q_pump_cmd
%   2. start pose, initial cap pressure and spool sign chosen by the base-workspace variable
%      move_09b2:  +1 boom-up from th_min (identical to lab08a stage4_run), -1 boom-down from th_max
%   3. spool opening fraction spool_frac_09b2 (1 = full stroke, as lab08a)
% All hydraulic parameters stay the lab08a values (dx300_params_lab08a.m and the lab08a InitFcn).
%
% Optional base-workspace variables (defaults reproduce the first 09b2 run):
%   src_model_09b2   lab08a model to copy: 'dx300_boom_hyd' (default) or 'dx300_boom_hyd_damped'
%   tag_09b2         suffix of the copy and of the exports, e.g. '_damped' (default '')
%   spool_ramp_09b2  s; > 0 adds a rate limiter after the spool step (as stage4_run section 1b),
%                    slew dS_max / spool_ramp_09b2. 0 (default) = step, no block added. Design choice.
here09 = fileparts(mfilename('fullpath'));
if ~exist('src_model_09b2','var'), src_model_09b2 = 'dx300_boom_hyd'; end
if ~exist('tag_09b2','var'), tag_09b2 = ''; end
if ~exist('spool_ramp_09b2','var'), spool_ramp_09b2 = 0; end
src = fullfile(here09,'..','lab08a',[src_model_09b2 '.slx']);
m = ['dx300_boom_hyd_09b2' tag_09b2];
if bdIsLoaded(m), close_system(m,0); end
copyfile(src, fullfile(here09,[m '.slx']), 'f'); fileattrib(fullfile(here09,[m '.slx']),'+w');
load_system(fullfile(here09,[m '.slx']));

init = get_param(m,'InitFcn');
init = [init newline ...
  '% --- Lab 09b2: selectable move (copy only) ---' newline ...
  'if ~exist(''move_09b2'',''var''), move_09b2 = 1; end            % +1 boom-up, -1 boom-down' newline ...
  'if ~exist(''spool_frac_09b2'',''var''), spool_frac_09b2 = 1; end  % fraction of dS_max (design choice)' newline ...
  'S9 = load(fullfile(here,''stage1_kin.mat''),''th'',''L'');' newline ...
  'if move_09b2 > 0' newline ...
  '  th0_09b2 = P.th_min; x0_09b2 = 0; p0_09b2 = p0_cap;            % as lab08a' newline ...
  'else' newline ...
  '  th0_09b2 = P.th_max;                                            % ESTIMATED end stop' newline ...
  '  x0_09b2 = min(interp1(S9.th(:), S9.L(:), th0_09b2, ''linear'', ''extrap'') - S9.L(1), P.stroke);  % rod extension at th_max' newline ...
  '  tau_g9 = P.m_boom*9.81*Lc0*cos(th0_09b2) + 9.81*([m_Dpt m_armpt m_tippt]*(pts_(:,1)*cos(th0_09b2)-pts_(:,2)*sin(th0_09b2)));' newline ...
  '  p0_09b2 = tau_g9/(interp1(th_tab, r_tab, th0_09b2, ''linear'', ''extrap'')*A_cap) + 101325;   % holding pressure at th_max, Pa abs' newline ...
  'end' newline ...
  'spool_after_09b2 = sign(move_09b2)*spool_frac_09b2*dS_max;'];
set_param(m,'InitFcn',init);

set_param([m '/Revolute Joint'],'PositionTargetValue','th0_09b2*180/pi');
set_param([m '/Cyl_eq'],'x0','x0_09b2','p0_A','p0_09b2');
set_param([m '/spool_cmd'],'After','spool_after_09b2');
if spool_ramp_09b2 > 0                                  % ramped spool: rate limiter between spool_cmd and Spool_SPS
    phs = get_param([m '/Spool_SPS'],'PortHandles'); delete_line(get_param(phs.Inport(1),'Line'));
    rl = add_block('simulink/Discontinuities/Rate Limiter',[m '/spool_rl'], ...
        'RisingSlewLimit','dS_max/spool_ramp_09b2','FallingSlewLimit','-dS_max/spool_ramp_09b2');
    add_line(m, get_param([m '/spool_cmd'],'PortHandles').Outport(1), get_param(rl,'PortHandles').Inport(1));
    add_line(m, get_param(rl,'PortHandles').Outport(1), phs.Inport(1));
end
set_param(m,'SimscapeLogLimitData','off');                 % default keeps only the last 5000 points: long runs lose the start of the move
ph = get_param([m '/pump_sat'],'PortHandles');
set_param(ph.Outport(1),'DataLogging','on','DataLoggingNameMode','Custom','DataLoggingName','Q_pump_cmd');

save_system(m, fullfile(here09,[m '.slx']));
close_system(m,0);
fprintf('built %s.slx (copy of lab08a/%s.slx + Q_pump_cmd logging + move selector, spool ramp %g s)\n', m, src_model_09b2, spool_ramp_09b2);
