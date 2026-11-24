function M = Mmat(q,p)
% Link mass matrix (4x4). Swing inertia M(1,1) changes with the pose.
M = zeros(4); M(1,1) = p.I1;
for i = 1:3
    [r,~,dr,dz,ph] = kin(q,p,i);
    M(1,1) = M(1,1) + p.m(i)*r^2 + p.Ic(i)*cos(ph)^2;
    a = [ones(1,i) zeros(1,3-i)];
    M(2:4,2:4) = M(2:4,2:4) + p.m(i)*(dr'*dr + dz'*dz) + p.Ic(i)*(a'*a);
end
end
