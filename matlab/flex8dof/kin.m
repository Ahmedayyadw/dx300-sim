function [r,z,dr,dz,phi] = kin(q,p,i)
% Planar point of link i (1 boom, 2 arm, 3 bucket): its COM, or the tip if i = 4
% (tip of bucket). Returns radius r, height z, and d/d[q2 q3 q4].
ph = cumsum(q(2:4)); r = p.r0; z = p.z0; dr = zeros(1,3); dz = zeros(1,3);
n = min(i,3);
for j = 1:n
    if j == i, l = p.lc(j); else, l = p.L(j); end
    r = r + l*cos(ph(j));  z = z + l*sin(ph(j));
    dr(1:j) = dr(1:j) - l*sin(ph(j));  dz(1:j) = dz(1:j) + l*cos(ph(j));
end
phi = ph(n);
end
