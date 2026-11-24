function p = dx300_params()
% DX300 front attachment parameters for the 8-DOF flexible-joint model.
% (PH) = placeholder -> replace with DX300 shop-manual / CAD values.
p.g  = 9.81;
p.r0 = 0.60;  p.z0 = 2.00;           % boom foot pin: radius from swing axis, height (PH)
p.L  = [6.25 3.10 1.60];             % boom (A->D), arm, bucket lengths [m] (arm/bucket PH)
p.lc = [2.90 1.30 0.70];             % pin -> centre of mass [m] (PH)
p.m  = [2330 1200 1100];             % masses [kg] (boom = DX300, others PH)
p.Ic = p.m .* p.L.^2 / 12;           % slender-link inertia about COM
p.I1 = 6.0e4;                        % upper structure about swing axis [kg m^2] (PH)

% Joint springs = oil stiffness  k = beta*(AA^2/VA + AB^2/VB) * (dL/dq)^2
p.beta = 1.5e9;  p.Vh = 0.003;       % bulk modulus [Pa], dead volume per side [m^3] (PH)
% cylinders: [bore, rod, stroke, moment arm dL/dq]   (boom = equivalent pair)
p.cyl = [ 0.140*sqrt(2) 0.095*sqrt(2) 1.45 0.80;
          0.150         0.105         1.60 0.70;     % arm (PH)
          0.135         0.095         1.10 0.50];    % bucket (PH)
k = zeros(1,4);
for i = 1:3
    AA = pi/4*p.cyl(i,1)^2;  AB = AA - pi/4*p.cyl(i,2)^2;
    VA = AA*p.cyl(i,3)/2 + p.Vh;  VB = AB*p.cyl(i,3)/2 + p.Vh;   % mid-stroke
    k(i+1) = p.beta*(AA^2/VA + AB^2/VB) * p.cyl(i,4)^2;
end
Dm = 150e-6/(2*pi); Vm = 0.002; ig = 150;            % swing motor + gearbox (PH)
k(1) = 2*p.beta*Dm^2/Vm * ig^2;
p.k = k;  p.K = diag(k);
p.J = diag([0.05*ig^2, 2*150*0.8^2, 150*0.7^2, 100*0.5^2]);  % actuator inertia (PH)
p.B = diag([2e3 5e3 3e3 1e3]);                                % actuator friction (PH)

% Initial pose, equilibrium, damping, actuator position loop
p.q0   = deg2rad([0 30 -80 -60])';
p.tau0 = Gvec(p.q0,p);
p.th0  = p.q0 + p.K \ p.tau0;
M0     = Mmat(p.q0,p);
p.D  = diag(2*0.05*sqrt(k(:).*diag(M0)));             % 5% spring damping
p.Kp = 0.5*p.K;
p.Kd = diag(2*0.9*sqrt(diag(p.Kp).*(diag(p.J)+diag(M0))));
p.ref = @(t) deg2rad([10*(t>=3), 5*(t>=1), 0, 0])';  % swing +10 deg @3 s, boom +5 deg @1 s
end
