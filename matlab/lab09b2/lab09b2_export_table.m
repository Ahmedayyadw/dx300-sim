function T = lab09b2_export_table(R, dt)
    % Lab 09b2: logs of one move on a uniform grid, from t = 0 to 0.5 s after arrival.
    tg = (0:dt:R.sum.t_reach_s + 0.5)';  tg = tg(tg <= R.t(end));
    f = {'theta','omega','tau','F','v','p_cap','p_rod','p_pump','Q_pump','Q_relief','Q_cap','Q_rod','S'};
    [tu, iu] = unique(R.t);
    T = table(tg, 'VariableNames', {'t'});
    for k = 1:numel(f), y = R.(f{k}); T.(f{k}) = interp1(tu, y(iu), tg); end
    T.in_window = tg >= 1.0 & tg <= R.sum.t_reach_s;
end
