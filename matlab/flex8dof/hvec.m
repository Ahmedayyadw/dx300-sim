function h = hvec(q,qd,p)
% Coriolis/centrifugal torques: h = Mdot*qd - 0.5*grad_q(qd'*M*qd)
e = 1e-6; Md = zeros(4); g2 = zeros(4,1);
for k = 1:4
    dq = zeros(4,1); dq(k) = e;
    dM = (Mmat(q+dq,p) - Mmat(q-dq,p))/(2*e);
    Md = Md + dM*qd(k);  g2(k) = qd'*dM*qd;
end
h = Md*qd - 0.5*g2;
end
