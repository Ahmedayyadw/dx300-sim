function G = Gvec(q,p)
% Gravity torque vector G = dVg/dq
G = zeros(4,1);
for i = 1:3
    [~,~,~,dz] = kin(q,p,i);
    G(2:4) = G(2:4) + p.m(i)*p.g*dz';
end
end
