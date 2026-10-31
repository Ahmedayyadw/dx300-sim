% Stage 3 - validate Multibody boom + front point masses vs analytic
m = 'dx300_boom_mb'; here = '/home/ahmed/ahmed/dx300-sim/matlab/lab08a';
load_system(fullfile(here,[m '.slx']))
p = get_param([m '/PS-Simulink Converter'],'PortHandles');
set_param(p.Outport(1),'DataLogging','on','DataLoggingNameMode','Custom','DataLoggingName','tau');
s = get_param([m '/theta_cmd'],'PortHandles');
set_param(s.Outport(1),'DataLogging','on','DataLoggingNameMode','Custom','DataLoggingName','theta');
out = sim(m);

tau = out.logsout.get('tau').Values;  th = out.logsout.get('theta').Values;
t = tau.Time;  q = interp1(th.Time, th.Data, t);
A = (P.th_max-P.th_min)/2;  w = 2*pi/20;  qdd = A*w^2*cos(w*t);

% boom (brick about A)
J_brick = P.m_boom*(P.L_boom^2+0.5^2)/12 + P.m_boom*Lc0^2;
tau_boom = P.m_boom*9.81*Lc0*cos(q);

% front point masses, positions in boom frame (arm fixed at phi_rel)
pts = [P.L_boom 0; P.L_boom+0.5*La*cos(phi_rel) 0.5*La*sin(phi_rel); ...
    P.L_boom+La*cos(phi_rel) La*sin(phi_rel)];
mp  = [m_Dpt m_armpt m_tippt];
xg  = pts(:,1)*cos(q') - pts(:,2)*sin(q');          % horizontal arm of each mass
tau_front = 9.81*(mp*xg)';
J_front = sum(mp(:).*sum(pts.^2,2));

tau_an = tau_boom + tau_front + (J_brick+J_front)*qdd;
k = t > 0.5;
err = max(abs(tau.Data(k)-tau_an(k)))/max(tau_an(k))*100;
fprintf('J_brick = %.0f, J_front = %.0f, J_total = %.0f kg m^2\n', J_brick, J_front, J_brick+J_front);
fprintf('max tau = %.1f kN m, max error = %.2f %%\n', max(tau.Data)/1e3, err);

figure; plot(rad2deg(q),tau.Data/1e3,'LineWidth',1.5); hold on
plot(rad2deg(q),tau_an/1e3,'--','LineWidth',1.5); grid on
xlabel('\theta [deg]'); ylabel('\tau [kN m]')
legend('Simscape Multibody','Analytic','Location','south')
title(sprintf('Boom + arm/bucket/payload (arm \\perp boom), max error %.2f %%', err))
exportgraphics(gcf, fullfile(here,'figs','stage3_validation_front.png'),'Resolution',200)
