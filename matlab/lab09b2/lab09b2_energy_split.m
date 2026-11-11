function R = lab09b2_energy_split(out, mv, t_spool, tol, P, A_cap, A_ann)
    % Lab 09b2: Simscape energy split of one move over the motion window (spool opening .. first arrival
    % within tol of the end stop). Used by lab09b2_run.m and lab09b2_explore.m.
    % All pressures gauge (the lab08a sensors are differential against an atmospheric reservoir).
    sl = out.simlog; lg = out.logsout;
    sv = @(node,unit) node.series.values(unit);
    t  = sl.p_pump.Pg.series.time;
    if t(1) > 1e-9, error('lab09b2:logTruncated','Simscape log starts at t = %.3f s, not 0: data history limit hit', t(1)); end
    ip = @(n) interp1(lg.get(n).Values.Time, lg.get(n).Values.Data, t);
    R.t = t;
    R.theta = ip('q'); R.omega = ip('w'); R.tau = ip('tau'); R.F = ip('F'); R.v = ip('v_rod');
    R.p_pump = sv(sl.p_pump.Pg,'Pa'); R.p_cap = sv(sl.p_cap.Pg,'Pa'); R.p_rod = sv(sl.p_rod.Pg,'Pa');
    R.Q_pump   = -sv(sl.Pump.q_B,'m^3/s');                      % pump discharge flow
    R.Q_cmd    = ip('Q_pump_cmd');                              % commanded flow (should equal Q_pump)
    R.Q_relief = sv(sl.Relief.q_A,'m^3/s');                     % into the relief valve
    R.Q_cap = sv(sl.Cyl_eq.chamber_A.mdot_A,'kg/s') ./ sv(sl.Cyl_eq.chamber_A.rho_A,'kg/m^3');   % into cap chamber
    R.Q_rod = sv(sl.Cyl_eq.chamber_B.mdot_A,'kg/s') ./ sv(sl.Cyl_eq.chamber_B.rho_A,'kg/m^3');   % into rod chamber
    R.S = sv(sl.DCV.S,'m');
    if mv.dir > 0, i1 = find(R.theta >= P.th_max - tol, 1); else, i1 = find(R.theta <= P.th_min + tol, 1); end
    if isempty(i1), error('lab09b2:noArrival','%s: end stop not reached within StopTime', mv.name); end
    t_reach = t(i1);
    w = t >= t_spool & t <= t_reach;                            % motion window
    I = @(y) trapz(t(w), y(w)) / 1e3;                           % kJ over the motion window
    P_pump = R.p_pump .* R.Q_pump;
    P_in_cap = R.p_cap .* R.Q_cap;  P_in_rod = R.p_rod .* R.Q_rod;   % hydraulic power into each chamber
    P_mech = R.tau .* R.omega;
    s = struct();
    s.move = string(mv.name);
    s.spool_frac = mv.dir*mv.frac;
    s.t_reach_s = t_reach;  s.t_move_s = t_reach - t_spool;
    s.theta0_deg = rad2deg(R.theta(1)); s.theta_reach_deg = rad2deg(R.theta(i1));
    s.E_pump_kJ = I(P_pump);                                    % pump hydraulic output
    s.E_relief_kJ = I(R.p_pump .* R.Q_relief);                  % dumped over the relief valve
    if mv.dir > 0                                               % pump feeds the cap side
        s.E_cyl_in_kJ = I(P_in_cap);  s.E_cyl_out_kJ = I(-P_in_rod);
    else                                                        % pump feeds the rod side
        s.E_cyl_in_kJ = I(P_in_rod);  s.E_cyl_out_kJ = I(-P_in_cap);
    end
    s.E_cyl_net_kJ = s.E_cyl_in_kJ - s.E_cyl_out_kJ;            % net hydraulic energy into the cylinder
    s.E_mech_kJ = I(P_mech);                                    % joint work, signed
    s.E_mech_abs_kJ = I(abs(P_mech));
    s.W_pos_kJ = I(max(P_mech,0));  s.W_neg_kJ = I(max(-P_mech,0));
    s.E_Fv_kJ = I(R.F .* R.v);                                  % rod work, should equal E_mech
    s.loss_pump_to_cyl_kJ = s.E_pump_kJ - s.E_cyl_in_kJ;        % relief + supply-side valve drop + line volume
    s.loss_supply_valve_kJ = s.loss_pump_to_cyl_kJ - s.E_relief_kJ;
    s.loss_cyl_to_mech_kJ = s.E_cyl_net_kJ - s.E_mech_kJ;       % oil compressibility (no friction, no leakage in the model)
    s.vol_cap_m3 = trapz(t(w), abs(R.Q_cap(w)));  s.vol_rod_m3 = trapz(t(w), abs(R.Q_rod(w)));
    s.vol_pump_m3 = trapz(t(w), R.Q_pump(w));     s.vol_relief_m3 = trapz(t(w), R.Q_relief(w));
    s.vol_kin_m3 = trapz(t(w), abs(R.v(w))) * (A_cap*(mv.dir>0) + A_ann*(mv.dir<0));   % A * swept stroke (Python definition)
    s.p_pump_mean_bar = mean_w(t(w), R.p_pump(w))/1e5;  s.p_pump_max_bar = max(R.p_pump(w))/1e5;
    s.p_cap_mean_bar = mean_w(t(w), R.p_cap(w))/1e5;    s.p_rod_mean_bar = mean_w(t(w), R.p_rod(w))/1e5;
    s.p_rod_min_bar = min(R.p_rod(w))/1e5;              s.p_cap_min_bar = min(R.p_cap(w))/1e5;
    s.v_mean_mps = mean_w(t(w), R.v(w));                s.v_peak_mps = max(abs(R.v(w)));
    s.relief_open_frac = trapz(t(w), double(R.p_pump(w) >= P.p_relief - 0.5e5)) / (t_reach - t_spool);
    pre = t <= t_spool;                                          % pump dead-headed before the spool opens
    s.E_pump_standby_kJ = trapz(t(pre), P_pump(pre))/1e3;
    s.Q_cmd_max_err = max(abs(R.Q_cmd - R.Q_pump));
    s.cav_cap_frac = trapz(t(w), double(R.p_cap(w) < -0.5e5)) / (t_reach - t_spool);   % share of the window below -0.5 bar gauge
    s.cav_rod_frac = trapz(t(w), double(R.p_rod(w) < -0.5e5)) / (t_reach - t_spool);
    s.p_cap_max_bar = max(R.p_cap(w))/1e5;
    s.Q_cyl_peak_Lmin = max(abs(R.v(w))) * (A_cap*(mv.dir>0) + A_ann*(mv.dir<0)) * 6e4;   % kinematic demand A*|v|
    R.sum = s;
end

function mval = mean_w(t, y), mval = trapz(t,y)/(t(end)-t(1)); end
