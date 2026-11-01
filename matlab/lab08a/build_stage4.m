% Stage 4 build - replace motion input of the Multibody boom with hydraulic actuation.
% Idempotent: rebuilds dx300_boom_hyd.slx from a fresh copy of dx300_boom_mb.slx.
% Run from anywhere:  run('build_stage4.m')   (needs SimHydraulics license, SimscapeFluids)
%
% Coupling (virtual work, equivalent single cylinder = 2 real cylinders):
%   rod velocity v = r(theta)*omega      (r from stage1_kin.mat)
%   joint torque tau = F_cyl*r(theta)
here = fileparts(mfilename('fullpath'));
src = 'dx300_boom_mb';  dst = 'dx300_boom_hyd';
if bdIsLoaded(dst), close_system(dst,0); end
copyfile(fullfile(here,[src '.slx']), fullfile(here,[dst '.slx']), 'f');
fileattrib(fullfile(here,[dst '.slx']), '+w');
run(fullfile(here,'dx300_params_lab08a.m'));                 % struct P
load_system('SimscapeFluids_lib'); load_system('fl_lib'); load_system('simulink');
load_system(fullfile(here,[dst '.slx']));
m = dst;

%% ---- InitFcn: also load r(theta) table from stage1 -----------------------
init = get_param(m,'InitFcn');
init = [init newline ...
  'S1 = load(fullfile(here,''stage1_kin.mat''),''th'',''r''); th_tab = S1.th(:)'';  r_tab = S1.r(:)'';  % moment arm r(theta), m (from ESTIMATED pins)' newline ...
  '% --- Stage 4 hydraulics ---' newline ...
  'A_cap = P.n_cyl*pi/4*P.bore^2;                 % OFFICIAL-derived [m^2] combined cap area of 2 cylinders' newline ...
  'A_ann = P.n_cyl*pi/4*(P.bore^2-P.rod^2);       % OFFICIAL-derived [m^2] combined annulus area' newline ...
  'rho_oil = 870;      % kg/m^3  ESTIMATED (typical mineral hydraulic oil)' newline ...
  'beta_oil = 1.5e9;   % Pa      ESTIMATED (typical effective bulk modulus)' newline ...
  'nu_oil = 46e-6;     % m^2/s   ESTIMATED (ISO VG46-ish, ~46 cSt)' newline ...
  'A_valve_max = 3e-4; % m^2     ESTIMATED (main spool / relief max area)' newline ...
  'dS_max = 5e-3;      % m       ESTIMATED (spool stroke)' newline ...
  'p_range_relief = 5e5; % Pa    ESTIMATED (relief opening range)' newline ...
  'V_dead = 1e-3;      % m^3     ESTIMATED (dead volume per chamber incl. hose/gallery, both cylinders)' newline ...
  '% static holding pressure at th_min (analytic gravity torque as in stage3_validate) -> consistent initial cap pressure' newline ...
  'q0_ = P.th_min; pts_ = [P.L_boom 0; P.L_boom+0.5*La*cos(phi_rel) 0.5*La*sin(phi_rel); P.L_boom+La*cos(phi_rel) La*sin(phi_rel)];' newline ...
  'tau_g0 = P.m_boom*9.81*Lc0*cos(q0_) + 9.81*([m_Dpt m_armpt m_tippt]*(pts_(:,1)*cos(q0_)-pts_(:,2)*sin(q0_)));   % N m' newline ...
  'p0_cap = tau_g0/(r_tab(1)*A_cap) + 101325;   % Pa abs' newline ...
  'spool_t = 1.0;      % s       spool step time (design choice)'];
set_param(m,'InitFcn',init);
% 'here' must exist inside InitFcn (original uses a hard-coded path) - it is defined in the original InitFcn line 1.

%% ---- Handles to blocks to reuse/copy -------------------------------------
nl = @(s) strrep(s,'\n',newline);
bPSS  = [m '/' nl('PS-Simulink\nConverter')];
bSPS  = [m '/' nl('Simulink-PS\nConverter')];
bSolv = [m '/' nl('Solver\nConfiguration')];
bJ    = [m '/Revolute Joint'];

%% ---- Remove old motion input chain ---------------------------------------
% Copy converters first (blocks are re-used as templates), then delete originals.
tmpl.pss = add_block(bPSS, [m '/tmpl_pss'],'MakeNameUnique','on');
tmpl.sps = add_block(bSPS, [m '/tmpl_sps'],'MakeNameUnique','on');
tmpl.sol = add_block(bSolv,[m '/tmpl_sol'],'MakeNameUnique','on');
for b = {[m '/theta_cmd'], bSPS, bPSS, [m '/Scope']}
    delete_block(b{1});
end

%% ---- Revolute joint: torque in, motion out -------------------------------
set_param(bJ, 'TorqueActuationMode','InputTorque', 'MotionActuationMode','ComputedMotion', ...
    'SenseTorqueForce','off', 'SensePosition','on', 'SenseVelocity','on', ...
    'PositionTargetSpecify','on', 'PositionTargetPriority','High', ...
    'PositionTargetValue','P.th_min*180/pi', 'PositionTargetValueUnits','deg');   % th_min OFFICIAL
jp = get_param(bJ,'PortHandles');
fprintf('Joint ports: L=%d R=%d\n', numel(jp.LConn), numel(jp.RConn));
jpc = get_param(bJ,'PortConnectivity');
for k=1:numel(jpc), fprintf('  %s @ %s\n', jpc(k).Type, mat2str(jpc(k).Position)); end

%% ---- helper-based block placement ---------------------------------------
X0 = 700; Y0 = 400;
IL  = 'fl_lib/Isothermal Liquid/';
SF  = 'SimscapeFluids_lib/Isothermal Liquid/';
pos = @(x,y,w,h) [x y x+w y+h];

%% ---- Mechanical-translational equivalent cylinder ------------------------
act = addb([SF 'Actuators/Double-Acting Actuator (IL)'], [m '/Cyl_eq'], pos(X0+300,Y0+150,90,60), { ...
    'piston_area_A','A_cap', ...          % OFFICIAL-derived (2 cyl)
    'piston_area_B','A_ann', ...          % OFFICIAL-derived (2 cyl)
    'stroke','P.stroke', ...              % OFFICIAL
    'dead_volume_A','V_dead', 'dead_volume_B','V_dead', ...   % ESTIMATED
    'x0','0', ...                         % rod retracted at th_min
    'p0_A','p0_cap', 'p0_A_unit','Pa', ...      % NB: library default unit is MPa
    ... % p0_A = holding pressure at th_min (valve closed)
    'hardstop_model','simscape.enum.hardstop.smooth' });
velsrc = addb('fl_lib/Mechanical/Mechanical Sources/Ideal Translational Velocity Source', [m '/V_src'], pos(X0+560,Y0+260,40,40), {});
fsens  = addb('fl_lib/Mechanical/Mechanical Sensors/Ideal Force Sensor', [m '/F_sens'], pos(X0+450,Y0+200,40,40), {});
tref1  = add_block(['fl_lib/Mechanical/Translational' newline 'Elements/Mechanical' newline 'Translational' newline 'Reference'], [m '/Tref1'], 'Position', pos(X0+560,Y0+340,30,30));
tref2  = add_block(['fl_lib/Mechanical/Translational' newline 'Elements/Mechanical' newline 'Translational' newline 'Reference'], [m '/Tref2'], 'Position', pos(X0+330,Y0+260,30,30));

%% ---- Hydraulic circuit ---------------------------------------------------
pump = addb([IL 'Sources/Flow Rate Source (IL)'], [m '/Pump'], pos(X0-100,Y0+60,50,40), { ...
    'source_type','foundation.enum.constant_controlled.controlled', ...   % input = volumetric flow (ramped, see below)
    'flow_type','foundation.enum.mass_volumetric_flow.volumetric' });
rlf  = addb([SF 'Valves & Orifices/Pressure Control Valves/Pressure Relief Valve (IL)'], [m '/Relief'], pos(X0-100,Y0-60,60,40), { ...
    'pressure_spec','fluids.isothermal_liquid.valves_orifices.enum.pressure_control_spec.pA', ...  % gauge
    'p_set_gauge','P.p_relief', 'p_set_gauge_unit','Pa', ...   % NB: default unit is MPa                          % OFFICIAL 330 bar
    'p_range','p_range_relief', 'p_range_unit','Pa', ...                          % ESTIMATED
    'area_max','A_valve_max' });                             % ESTIMATED
dcv  = addb([SF 'Valves & Orifices/Directional Control Valves/4-Way 3-Position Directional Valve (IL)'], [m '/DCV'], pos(X0+100,Y0+100,80,60), { ...
    'area_max','A_valve_max', ...                            % ESTIMATED
    'del_S_max','dS_max' });                                 % ESTIMATED
res1 = addb([IL 'Elements/Reservoir (IL)'], [m '/Res_pump'],  pos(X0-170,Y0+120,30,30), {});
res2 = addb([IL 'Elements/Reservoir (IL)'], [m '/Res_relief'],pos(X0+20,Y0-60,30,30), {});
res3 = addb([IL 'Elements/Reservoir (IL)'], [m '/Res_T'],     pos(X0+190,Y0+200,30,30), {});
prop = addb([IL 'Utilities/Isothermal Liquid Properties (IL)'], [m '/Oil'], pos(X0-30,Y0-130,60,40), { ...
    'rho_L_atm','rho_oil', 'beta_L_atm','beta_oil', 'nu_atm','nu_oil', ...   % ESTIMATED
    'bulk_modulus_model','foundation.enum.bulk_modulus_model.const' });
% pump-line volume: without it the dead-headed pump node is incompressible (flow into zero volume) -> stiff DAE
Vln  = addb([IL 'Elements/Constant Volume Chamber (IL)'], [m '/V_pumpline'], pos(X0-30,Y0-50,40,30), {'volume','2e-3'});   % 2 L ESTIMATED (pump line + manifold)
psP  = addb([IL 'Sensors/Pressure Sensor (IL)'], [m '/p_pump'], pos(X0-30,Y0+20,40,40), {});
psA  = addb([IL 'Sensors/Pressure Sensor (IL)'], [m '/p_cap'],  pos(X0+230,Y0+60,40,40), {});
psB  = addb([IL 'Sensors/Pressure Sensor (IL)'], [m '/p_rod'],  pos(X0+230,Y0+220,40,40), {});
resP1 = addb([IL 'Elements/Reservoir (IL)'], [m '/Res_ref1'], pos(X0+30,Y0+20,30,30), {});
resP2 = addb([IL 'Elements/Reservoir (IL)'], [m '/Res_ref2'], pos(X0+290,Y0+60,30,30), {});
resP3 = addb([IL 'Elements/Reservoir (IL)'], [m '/Res_ref3'], pos(X0+290,Y0+220,30,30), {});

solF = add_block(tmpl.sol, [m '/Solver_fluid'], 'Position', pos(X0-100,Y0-130,40,40));
% NOTE: fluid + translational-mechanical (via the actuator) form ONE physical network -> one solver config.
% The multibody network keeps its own original Solver Configuration.
solFh = get_param(solF,'PortHandles'); %#ok<NASGU>

%% ---- Pump flow command: Q_pump (OFFICIAL 494 L/min) ramped over 0.2 s -----
% (a step into a dead-headed line makes the initial-condition solve fail; ramp time is a design choice)
rmp  = addb('simulink/Sources/Ramp', [m '/pump_ramp'], pos(X0-260,Y0+60,40,30), {'slope','P.Q_pump/0.2'});
sat  = addb('simulink/Discontinuities/Saturation', [m '/pump_sat'], pos(X0-210,Y0+60,40,30), {'UpperLimit','P.Q_pump','LowerLimit','0'});
cQ   = add_block(tmpl.sps, [m '/Q_SPS'], 'Position', pos(X0-160,Y0+60,50,30)); set_param(cQ,'Unit','m^3/s');

%% ---- Spool command -------------------------------------------------------
stp  = addb('simulink/Sources/Step', [m '/spool_cmd'], pos(X0-60,Y0+160,40,30), { ...
    'Time','spool_t', 'Before','0', 'After','dS_max' });     % step to full open at 1 s
cS   = add_block(tmpl.sps, [m '/Spool_SPS'], 'Position', pos(X0,Y0+160,50,30));
set_param(cS,'Unit','m');

%% ---- Signal chain: q, w -> r(q) -> v ;  F -> tau --------------------------
qConv = add_block(tmpl.pss,[m '/q_PS'], 'Position', pos(X0,Y0+400,50,30)); set_param(qConv,'Unit','rad');
wConv = add_block(tmpl.pss,[m '/w_PS'], 'Position', pos(X0,Y0+450,50,30)); set_param(wConv,'Unit','rad/s');
lut   = addb('simulink/Lookup Tables/1-D Lookup Table', [m '/r_of_theta'], pos(X0+90,Y0+400,60,40), { ...
    'BreakpointsForDimension1','th_tab', 'Table','r_tab', 'ExtrapMethod','Clip', 'InterpMethod','Linear point-slope'});
mulv  = addb('simulink/Math Operations/Product', [m '/v_eq'], pos(X0+190,Y0+420,30,40), {});
vConv = add_block(tmpl.sps,[m '/v_SPS'], 'Position', pos(X0+260,Y0+420,50,30)); set_param(vConv,'Unit','m/s');
fConv = add_block(tmpl.pss,[m '/F_PS'], 'Position', pos(X0+260,Y0+500,50,30)); set_param(fConv,'Unit','N');
mult  = addb('simulink/Math Operations/Product', [m '/tau_eq'], pos(X0+340,Y0+500,30,40), {});
tConv = add_block(tmpl.sps,[m '/tau_SPS'], 'Position', pos(X0+410,Y0+500,50,30)); set_param(tConv,'Unit','N*m');
pPc = add_block(tmpl.pss,[m '/pp_PS'],'Position',pos(X0+60,Y0+560,50,30)); set_param(pPc,'Unit','Pa');
pAc = add_block(tmpl.pss,[m '/pA_PS'],'Position',pos(X0+60,Y0+600,50,30)); set_param(pAc,'Unit','Pa');
pBc = add_block(tmpl.pss,[m '/pB_PS'],'Position',pos(X0+60,Y0+640,50,30)); set_param(pBc,'Unit','Pa');
xc  = add_block(tmpl.pss,[m '/x_PS'], 'Position',pos(X0+60,Y0+680,50,30)); set_param(xc,'Unit','m');


%% ---- Wiring --------------------------------------------------------------
% Port maps (verified via PortConnectivity; labels inferred from block layout and checked in stage4_run):
%  Joint: L1 base, L2 torque in, R1 follower, R2 q, R3 w
%  Actuator (probed): L1 mech C(case), L2 fluid A(cap), R1 p out (unused), R2 mech R(rod), R3 fluid B(rod side)
%  DCV: L1 spool S, L2 P, L3 T, R1 A, R2 B
%  Flow src (controlled, probed): L1 A suction, R1 B discharge, L2 signal in    Pressure sensor: L1 port, R1 ref, R2 signal
%  Force sensor: L1,R1 mech, R2 signal out   Velocity source: L1,R2 mech, R1 signal in
%  PS-Simulink: LConn1 in, Outport1.  Simulink-PS: Inport1, RConn1 out.
c = @(a,ta,na,b,tb,nb) add_line(m, P_(a,ta,na), P_(b,tb,nb), 'autorouting','on');
% multibody side
c(bJ,'LConn',2, tConv,'RConn',1);                 % tau -> joint torque input
c(bJ,'RConn',2, qConv,'LConn',1);                 % q sensed
c(bJ,'RConn',3, wConv,'LConn',1);                 % w sensed
% Simulink side
add_line(m, P_(qConv,'Outport',1), P_(lut,'Inport',1), 'autorouting','on');
add_line(m, P_(lut,'Outport',1),   P_(mulv,'Inport',1),'autorouting','on');
add_line(m, P_(wConv,'Outport',1), P_(mulv,'Inport',2),'autorouting','on');
add_line(m, P_(mulv,'Outport',1),  P_(vConv,'Inport',1),'autorouting','on');
add_line(m, P_(fConv,'Outport',1), P_(mult,'Inport',1),'autorouting','on');
add_line(m, P_(lut,'Outport',1),   P_(mult,'Inport',2),'autorouting','on');
add_line(m, P_(mult,'Outport',1),  P_(tConv,'Inport',1),'autorouting','on');
add_line(m, P_(stp,'Outport',1),   P_(cS,'Inport',1),   'autorouting','on');
% translational chain: rod -> force sensor -> velocity source -> ref ; case -> ref
c(vConv,'RConn',1, velsrc,'RConn',1);
c(act,'RConn',2, fsens,'LConn',1);
c(fsens,'RConn',1, velsrc,'LConn',1);
c(velsrc,'RConn',2, tref1,'LConn',1);
c(act,'LConn',1, tref2,'LConn',1);
c(fsens,'RConn',2, fConv,'LConn',1);
% hydraulics
c(pump,'LConn',1, res1,'LConn',1);                % suction from tank (A)
add_line(m,P_(rmp,'Outport',1),P_(sat,'Inport',1),'autorouting','on'); add_line(m,P_(sat,'Outport',1),P_(cQ,'Inport',1),'autorouting','on'); c(cQ,'RConn',1,pump,'LConn',2);
c(pump,'RConn',1, rlf,'LConn',1);                 % pump node = relief inlet
c(rlf,'RConn',1, res2,'LConn',1);
c(pump,'RConn',1, dcv,'LConn',2);                 % pump node -> P
c(res3,'LConn',1, dcv,'LConn',3);                 % T -> tank
c(dcv,'RConn',1, act,'LConn',2);                  % A -> cap side
c(dcv,'RConn',2, act,'RConn',3);                  % B -> rod side
c(cS,'RConn',1, dcv,'LConn',1);                   % spool command
c(Vln,'LConn',1, rlf,'LConn',1);                  % line volume at pump node
c(prop,'RConn',1, rlf,'LConn',1);                 % fluid properties on circuit
% pressure sensors (ref port to tank)
c(psP,'LConn',1, rlf,'LConn',1); c(psP,'RConn',1, resP1,'LConn',1); c(psP,'RConn',2, pPc,'LConn',1);
c(psA,'LConn',1, act,'LConn',2); c(psA,'RConn',1, resP2,'LConn',1); c(psA,'RConn',2, pAc,'LConn',1);
c(psB,'LConn',1, act,'RConn',3); c(psB,'RConn',1, resP3,'LConn',1); c(psB,'RConn',2, pBc,'LConn',1);
% solver configs
c(solF,'RConn',1, res1,'LConn',1);
% logging
logs = {qConv,'Outport','q';  wConv,'Outport','w'; fConv,'Outport','F'; pPc,'Outport','p_pump'; ...
        pAc,'Outport','p_cap'; pBc,'Outport','p_rod'};
for i=1:size(logs,1)
    ph = P_(logs{i,1},'Outport',1);
    set_param(ph,'DataLogging','on','DataLoggingNameMode','Custom','DataLoggingName',logs{i,3});
end
set_param(P_(mult,'Outport',1),'DataLogging','on','DataLoggingNameMode','Custom','DataLoggingName','tau');
set_param(P_(mulv,'Outport',1),'DataLogging','on','DataLoggingNameMode','Custom','DataLoggingName','v_rod');
delete_block(xc);   % x_PS unused (actuator p-out port not wired)
for blk = {pPc,pAc,pBc}
    tb = add_block('simulink/Sinks/Terminator',[m '/term_' get_param(blk{1},'Name')],'Position',tpos(get_param(blk{1},'Position')));
    add_line(m, P_(blk{1},'Outport',1), P_(tb,'Inport',1));
end
set_param(m,'SignalLogging','on','SignalLoggingName','logsout','StopTime','10');

% remove template copies
delete_block(tmpl.pss); delete_block(tmpl.sps); delete_block(tmpl.sol);

save_system(m, fullfile(here,[dst '.slx']));
fprintf('Stage-4 skeleton blocks placed (connections added in part 2).\n');

function h = addb(path, name, pos, params)
    h = add_block(path, name, 'Position', pos);
    for i=1:2:numel(params), set_param(h, params{i}, params{i+1}); end
end

function h = P_(blk, type, n)
    ph = get_param(blk,'PortHandles'); h = ph.(type)(n);
end

function r = tpos(p)
    r = [p(3)+30 p(2)+5 p(3)+50 p(2)+25];
end
