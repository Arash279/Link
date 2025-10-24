%% ===== Robust FRD extraction & plotting with channel/validity checks =====
out.who;

% --- 1) 取频响矩阵与Info ---
H = out.get('frd_out');                 % 51 x 100 complex double（你当前情况）
S = out.get('simout1');                 % struct，含 Info/时域监控
fprintf('H: class=%s, size=%s, isreal=%d\n', class(H), mat2str(size(H)), isreal(H));

% --- 2) 频率向量（优先从 Info，自动判单位） ---
fHz = [];
try
    if isfield(S,'Info') && isfield(S.Info,'SinestreamFreq') && isa(S.Info.SinestreamFreq,'timeseries')
        fraw = S.Info.SinestreamFreq.Data(:);
        fprintf('来自 SinestreamFreq：N=%d, [min,max]=[%.3g, %.3g]\n', numel(fraw), min(fraw), max(fraw));
        fraw = fraw(isfinite(fraw) & fraw>0);
        fraw = unique(fraw,'stable');  % 保序去重
        if max(fraw) > 6.9e8, fHz = fraw/(2*pi); else, fHz = fraw; end
    elseif isfield(S,'Info') && isfield(S.Info,'Estimation') && isfield(S.Info.Estimation,'Frequencies')
        fraw = S.Info.Estimation.Frequencies(:);
        fprintf('来自 Estimation.Frequencies：N=%d, [min,max]=[%.3g, %.3g]\n', numel(fraw), min(fraw), max(fraw));
        fraw = fraw(isfinite(fraw) & fraw>0);
        if max(fraw) > 6.9e8, fHz = fraw/(2*pi); else, fHz = fraw; end
    end
catch ME
    fprintf('[警告] 读取 Info 失败：%s\n', ME.message);
end

if isempty(fHz)
    % 兜底：按面板设置（10~110 MHz，对数，点数=H第二维）
    fHz = logspace(log10(10), log10(110e6), size(H,2)).';
    fprintf('[兜底] 用 10~110MHz 生成频率：N=%d\n', numel(fHz));
end
Nw = numel(fHz);  w = 2*pi*fHz;
fprintf('频率（Hz）最终：N=%d, [min,max]=[%.3g, %.3g]\n', Nw, min(fHz), max(fHz));

% --- 3) 识别“有效频点”（任一通道非零即有效），丢弃全零频点 ---
if size(H,2)~=Nw && size(H,1)==Nw
    % 万一你的频率在第一维
    H = H.';  % 转置成 (通道 x 频点)
end
nonzero_by_freq = any(abs(H)>0, 1);
valid_idx = find(nonzero_by_freq);
fprintf('有效频点数=%d / %d\n', numel(valid_idx), Nw);

if numel(valid_idx) < Nw
    fprintf('[提示] 检测到未扫完的频点：将自动丢弃 %d 个全零频点（通常是 Stop time 不足或 start/stop 提前为0）。\n', Nw-numel(valid_idx));
end

H_valid  = H(:, valid_idx);
fHz_valid = fHz(valid_idx);
w_valid   = 2*pi*fHz_valid;
fprintf('H_valid size=%s, 频率范围=[%.3g, %.3g] Hz\n', mat2str(size(H_valid)), min(fHz_valid), max(fHz_valid));

% --- 4) 在 51 个通道里自动选择“正确通道” ---
% 规则：先按“有效非零点计数”排序，再按“能量(sum|H|^2)”排序
nonzero_counts = sum(abs(H_valid)>0, 2);
energy         = sum(abs(H_valid).^2,  2);

[~, idx1] = sort(nonzero_counts, 'descend');
[~, idx2] = sort(energy,         'descend');

% 取交集里排名靠前的候选
candidates = unique([idx1(1:min(10,end)); idx2(1:min(10,end))],'stable');
fprintf('候选通道(前10)：'); disp(candidates.');

% 选出“非零点最多且能量也高”的第一个
best_k = candidates(1);
fprintf('选中通道 k=%d；nonzero=%d，energy=%.3g\n', best_k, nonzero_counts(best_k), energy(best_k));

% 预览该通道前10个点
disp('该通道 |H|(1:10) 预览：'); disp(abs(H_valid(best_k,1:min(10,size(H_valid,2)))));

% --- 5) 构造 FRD 并绘图（默认 H=阻抗；若是导纳则改成 1./resp） ---
resp = reshape(H_valid(best_k, :), [1 1 numel(fHz_valid)]);
Zfrd  = frd(resp, w_valid);   % 若你的设置是电压激励/测电流 => Zfrd = frd(1./resp, w_valid);

opt = bodeoptions; opt.FreqUnits='Hz'; opt.MagUnits='abs';
opt.XLim = [min(fHz_valid) max(fHz_valid)];
figure; bode(Zfrd, opt); grid on;
title(sprintf('Port Impedance |Z(j\\omega)|  (channel %d, valid %d/%d)', best_k, numel(valid_idx), Nw));

% 自定义幅相（只用有效频点）
[G,wr] = freqresp(Zfrd); Z = squeeze(G); fplot = wr/(2*pi);
figure;
subplot(2,1,1); semilogx(fplot, abs(Z)); grid on;
xlabel('Frequency (Hz)'); ylabel('|Z| (Ohm)'); title('Impedance Magnitude (valid only)');
subplot(2,1,2); semilogx(fplot, unwrap(angle(Z))*180/pi); grid on;
xlabel('Frequency (Hz)'); ylabel('Phase (deg)'); title('Impedance Phase (valid only)');
