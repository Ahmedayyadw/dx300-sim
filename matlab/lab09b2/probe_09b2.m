% Lab 09b2 probe - list what the Simscape log of the copied model holds (boom-up run).
here09 = fileparts(mfilename('fullpath')); cd(here09);
run(fullfile(here09,'lab09b2_build.m'));
m = 'dx300_boom_hyd_09b2';
load_system(fullfile(here09,[m '.slx']));
move_09b2 = 1; spool_frac_09b2 = 1;
set_param(m,'StopTime','9');
tic; out = sim(m); fprintf('sim wall time %.1f s\n', toc);
lg = out.logsout;
fprintf('logsout: %s\n', strjoin(lg.getElementNames', ', '));
sl = out.simlog;
for blk = {'Pump','Relief','DCV','Cyl_eq','V_pumpline','p_pump','p_rod','F_sens','Oil'}
    fprintf('--- simlog.%s ---\n', blk{1}); dumpnode(sl.(blk{1}), blk{1}, 0);
end
pr = lg.get('p_rod').Values; pp = lg.get('p_pump').Values; pc = lg.get('p_cap').Values;
fprintf('p_rod at t=0.5: %.1f Pa   p_pump at t=0.9: %.1f Pa   p_cap at t=0: %.1f Pa (p0_cap abs = %.1f)\n', ...
    interp1(pr.Time,pr.Data,0.5), interp1(pp.Time,pp.Data,0.9), pc.Data(1), evalin('base','p0_cap'));
S = load(fullfile(here09,'..','lab08a','stage1_kin.mat'));
fprintf('stage1_kin: th %.2f..%.2f deg, L %.4f..%.4f m (L(end)-L(1) = %.4f), n = %d\n', rad2deg(S.th(1)), rad2deg(S.th(end)), S.L(1), S.L(end), S.L(end)-S.L(1), numel(S.th));
close_system(m,0);

function dumpnode(n, name, depth)
    if depth > 2, return; end
    ids = n.childIds;
    for k = 1:numel(ids)
        c = n.(ids{k});
        if c.numChildren == 0
            try
                s = c.series; u = s.unit; v = s.values(u);
                fprintf('%s%s.%s [%s] n=%d  first %.4g  max %.4g  min %.4g\n', repmat('  ',1,depth), name, ids{k}, u, numel(v), v(1), max(v), min(v));
            catch
                fprintf('%s%s.%s (no series)\n', repmat('  ',1,depth), name, ids{k});
            end
        else
            dumpnode(c, [name '.' ids{k}], depth+1);
        end
    end
end
