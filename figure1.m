% 读取 CSV 文件
data = readtable('D:\Desktop\Data\exp_10.csv');

% 提取列
freq = data.Freq;
zabs = data.Zabs;
phase = data.Phase;

% 目标频率
f1 = 39955.3;
f2 = 250198;

% 找到最接近这两个频率的索引
[~, idx1] = min(abs(freq - f1));
[~, idx2] = min(abs(freq - f2));

% 绘制图形
figure;

% ===============================
% 幅频特性（Zabs vs Freq）
% ===============================
subplot(2,1,1);
semilogx(freq, zabs, 'b', 'LineWidth', 1.5); hold on;

% 标记两个频率点
plot(freq(idx1), zabs(idx1), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(freq(idx2), zabs(idx2), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);

% 添加文字标签
text(freq(idx1), zabs(idx1), sprintf('  f = %.1f Hz', f1), 'Color', 'r', 'FontSize', 10, 'VerticalAlignment','bottom');
text(freq(idx2), zabs(idx2), sprintf('  f = %.1f Hz', f2), 'Color', 'r', 'FontSize', 10, 'VerticalAlignment','bottom');

grid on;
xlabel('Frequency (Hz)');
ylabel('|Z| (Ohm)');
title('Amplitude Response (Impedance Magnitude)');
xlim([min(freq) max(freq)]);

% ===============================
% 相频特性（Phase vs Freq）
% ===============================
subplot(2,1,2);
semilogx(freq, phase, 'r', 'LineWidth', 1.5); hold on;

% 标记两个频率点
plot(freq(idx1), phase(idx1), 'bo', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(freq(idx2), phase(idx2), 'bo', 'MarkerSize', 8, 'LineWidth', 1.5);

% 添加文字标签
text(freq(idx1), phase(idx1), sprintf('  f = %.1f Hz', f1), 'Color', 'b', 'FontSize', 10, 'VerticalAlignment','bottom');
text(freq(idx2), phase(idx2), sprintf('  f = %.1f Hz', f2), 'Color', 'b', 'FontSize', 10, 'VerticalAlignment','bottom');

grid on;
xlabel('Frequency (Hz)');
ylabel('Phase (°)');
title('Phase Response');
xlim([min(freq) max(freq)]);

% 总标题
sgtitle('Motor Impedance Frequency Response with Key Points');
