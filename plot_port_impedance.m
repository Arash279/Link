function plot_port_impedance(out)
    % === 1) 频率向量（从 FRE 的 Sinestream 记录里提取，单位 Hz）===
    if ~isfield(out,'simout1') || ~isfield(out.simout1,'Info') || ...
       ~isfield(out.simout1.Info,'SinestreamFreq')
        error('没找到 out.simout1.Info.SinestreamFreq');
    end
    fts = out.simout1.Info.SinestreamFreq;   % 1x1 double timeseries
    fraw = fts.Data(:);
    % 取“频率跳变点”的独立频率，并去掉 0/NaN
    fHz  = fraw([true; diff(fraw)~=0]);
    fHz  = fHz(isfinite(fHz) & fHz>0);
    % 保留原顺序的去重
    [~,idx] = unique(fHz,'stable'); 
    fHz = fHz(sort(idx));
    Nw  = numel(fHz);
    w   = 2*pi*fHz;  % rad/s 供 frd 使用

    % === 2) 频响矩阵 H（你用 frd 端口接到 To Workspace 得到的 51x100 complex double）===
    if isfield(out,'frd_out')
        H = out.frd_out;          % 例如 51x100 complex double
    else
        error('没找到 out.frd_out（检查 To Workspace 变量名）');
    end

    % === 3) 将 H 变为 1x1xN 的 SISO 频响张量 resp ===
    sz = size(H);
    if ndims(H) == 2
        if sz(2) == Nw              % 形如 M x Nw
            resp = reshape(H(1,:), [1 1 Nw]);   % 取第1行（单端口）
        elseif sz(1) == Nw          % 形如 Nw x M
            resp = reshape(H(:,1).', [1 1 Nw]); % 取第1列
        else
            error('H 尺寸与频点数不匹配：size(H)=%s, Nw=%d', mat2str(sz), Nw);
        end
    elseif ndims(H) == 3 && sz(3) == Nw
        resp = H(1,1,:);            % 多输入多输出时取(1,1,:)通道
    else
        error('无法识别 H 的维度：size(H)=%s', mat2str(sz));
    end

    % === 4) 得到阻抗 Z(jw) 的 FRD 对象并绘图 ===
    % 默认：电流激励、测电压 => H 即 Z；若相反则改为 Zfrd = frd(1./resp, w);
    Zfrd = frd(resp, w);

    opt = bodeoptions;
    opt.FreqUnits = 'Hz';
    opt.MagUnits  = 'abs';   % 用欧姆显示；想看 dB 改 'dB'
    opt.XLim      = [min(fHz) max(fHz)];

    figure; bode(Zfrd, opt); grid on; title('Port Impedance |Z(j\omega)|');

    % 同时给一张自定义幅相图
    [G,wr] = freqresp(Zfrd);
    Z = squeeze(G);  fplot = wr/(2*pi);
    figure;
    subplot(2,1,1); semilogx(fplot, abs(Z)); grid on;
    xlabel('Frequency (Hz)'); ylabel('|Z| (Ohm)'); title('Impedance Magnitude');
    subplot(2,1,2); semilogx(fplot, unwrap(angle(Z))*180/pi); grid on;
    xlabel('Frequency (Hz)'); ylabel('Phase (deg)'); title('Impedance Phase');
end



