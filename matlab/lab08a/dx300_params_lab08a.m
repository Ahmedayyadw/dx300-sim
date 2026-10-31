% DX300 boom parameters - Lab 08a
% Frame: origin pin A, x fwd, y up, theta = angle(A->D) from horizontal
P.theta_ref = deg2rad(-6.3);         % ESTIMATED (drawing)

% Pins at theta_ref [m]
P.A = [0 0];                         % definition
P.B = [0.571 -0.695];                % ESTIMATED (drawing), fixed to frame
P.C = [2.629  0.850];                % ESTIMATED (drawing), on boom
P.D = [6.208 -0.682];                % ESTIMATED (drawing), on boom

% Boom
P.L_boom = 6.245;                    % OFFICIAL [m]
P.m_boom = 2330;                     % OFFICIAL [kg]

% Cylinders (x2)
P.n_cyl   = 2;                       % OFFICIAL
P.bore    = 0.140;                   % OFFICIAL [m]
P.rod     = 0.095;                   % OFFICIAL [m]
P.stroke  = 1.450;                   % OFFICIAL [m]
P.L_closed= 2.010;                   % OFFICIAL [m]
P.m_cyl   = 260;                     % OFFICIAL [kg each]

% Pin diameters [m]
P.dA=0.105; P.dB=0.090; P.dC=0.090; P.dD=0.100;   % OFFICIAL

% Hydraulics
P.p_relief = 330e5;                  % OFFICIAL [Pa] (330-335 bar)
P.p_boost  = 350e5;                  % OFFICIAL [Pa] (350-360 bar)
P.Q_pump   = 2*247/60000;            % OFFICIAL [m^3/s] (494 L/min)

% End stops
P.th_min = deg2rad(-47.2);           % OFFICIAL (per param_audit)
P.th_max = deg2rad(60.4);            % OFFICIAL (per param_audit)
