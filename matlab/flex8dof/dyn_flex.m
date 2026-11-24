function xd = dyn_flex(t,x,p)
% 8-DOF flexible-joint model, x = [q; qd; th; thd] (16 states)
q = x(1:4); qd = x(5:8); th = x(9:12); thd = x(13:16);
s   = p.K*(th-q) + p.D*(thd-qd);                        % spring + damper torque
tau = p.tau0 + p.Kp*(p.th0 + p.ref(t) - th) - p.Kd*thd; % actuator position loop
qdd  = Mmat(q,p) \ (s - hvec(q,qd,p) - Gvec(q,p));
thdd = p.J \ (tau - p.B*thd - s);
xd = [qd; qdd; thd; thdd];
end
