function xd = dyn_rigid(t,x,p)
% Rigid 4-DOF model (K -> inf, so th = q), x = [q; qd]
q = x(1:4); qd = x(5:8);
tau = p.tau0 + p.Kp*(p.q0 + p.ref(t) - q) - p.Kd*qd;
qdd = (Mmat(q,p) + p.J) \ (tau - p.B*qd - hvec(q,qd,p) - Gvec(q,p));
xd = [qd; qdd];
end
