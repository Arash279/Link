function plot_port_Z(out)
    % ==== 0) 取频响矩阵 H ====
    if isfield(out,'frd_out')
        H = out.frd_out;             % e.g., 51x100 complex double
    elseif isfield(out,'simout') && isfield(out.simout,'signals')
        H = out.simout.signals.values;
    else
        error('没找到频响矩阵：请确认 To Workspace 的变量名（如 out.frd_out）。');
    end
    sz = size(H);

    % ==== 1) 尝试找频率（Hz），按常见路径逐一尝试 ====
    fHz = try_get_freq_hz(out);
    Nw  = [];
    if ~isempty(fHz)
        fHz = fHz(:);
        fHz = fHz(isfinite(fHz) & fHz>0);
        [~,idx] = unique(fHz,'stable');
        fHz = fHz(sort(idx));
        Nw = numel(fHz);
    end

    % ==== 2) 若没找到频率，就按扫频设置重建 ====
    if isempty(Nw)
        % 用矩阵维度猜测频点数
        cands = sz; cands(cands<4) = [];  % 忽略太小的维度
        if isempty(cands), cands = sz; end
        Nw_guess = cands(end);            % 常见为第二维
        % 你的 FRE 面板：logspace(log10(10), log10(110e6), 100)
        fHz = logspace(log10(10), log10(110e6), Nw_guess).';
        Nw  = numel(fHz);
        warning('未在 out 里找到频率，按 10~110MHz 的对数扫频（%d 点）重建。', Nw);
    end
    w = 2*pi*fHz; % rad/s

    % ==== 3) 把 H 变成 1x1xNw 的 SISO 频响 ====
    nd = ndims(H);
    if nd==2
        if sz(2)==Nw
            chan = 1;                  % 如需其它通道改这里
            resp = reshape(H(chan,:), [1 1 Nw]);
        elseif sz(1)==Nw
            chan = 1;
            resp = reshape(H(:,chan).', [1 1 Nw]);
        else
            error('H 尺寸与频点数不匹配：size(H)=%s, Nw=%d', mat2str(sz), Nw);
        end
    elseif nd==3 && sz(3)==Nw
        resp = H(1,1,:);               % 多输入多输出时取(1,1,:)；需其它通道自行修改
    else
        error('未识别的 H 维度：size(H)=%s', mat2str(sz));
    end

    % ==== 4) 生成 FRD；默认 H=阻抗（电流激励、测电压）====
    Zfrd = frd(resp, w);               % 若是电压激励/测电流，则用 frd(1./resp, w)

    % ==== 5) 画 Bode（欧姆 & Hz）====
    opt = bodeoptions; opt.FreqUnits='Hz'; opt.MagUnits='abs'; opt.XLim=[min(fHz) max(fHz)];
    figure; bode(Zfrd, opt); grid on; title('Port Impedance |Z(j\omega)|');

    % 自定义幅相图
    [G,wr] = freqresp(Zfrd); Z = squeeze(G); fplot = wr/(2*pi);
    figure;
    subplot(2,1,1); semilogx(fplot, abs(Z)); grid on;
      xlabel('Frequency (Hz)'); ylabel('|Z| (Ohm)'); title('Impedance Magnitude');
    subplot(2,1,2); semilogx(fplot, unwrap(angle(Z))*180/pi); grid on;
      xlabel('Frequency (Hz)'); ylabel('Phase (deg)'); title('Impedance Phase');
end

function fHz = try_get_freq_hz(out)
    fHz = [];
    % 1) Simulink Control Design新旧版本常见路径
    try
        fts = out.simout1.Info.SinestreamFreq;         % timeseries
        fHz = fts.Data(:);
        return;
    catch, end
    try
        f = out.simout1.Info.Estimation.Frequencies;   % 可能是 Hz 或 rad/s
        f = f(:);
        % 粗判单位：>~7e8 视为 rad/s
        if max(f) > 6.9e8, fHz = f/(2*pi); else, fHz = f; end
        return;
    catch, end
    try
        f = out.simout1.Info.Frequencies;              % 兜底
        f = f(:);
        if max(f) > 6.9e8, fHz = f/(2*pi); else, fHz = f; end
        return;
    catch, end
end
