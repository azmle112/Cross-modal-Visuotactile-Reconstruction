"""绘制频谱图"""

import numpy as np
import matplotlib.pyplot as plt

fake = np.load("./generated_data/generated_spec.npy")
real = np.load("./generated_data/batch_spec.npy")


def draw(s, flag, k):
    max_value = np.max(s)
    min_value = np.min(s)
    if flag == real:
        magnitude_spectrum = (s + 1) / 2 * (max_value - min_value) + min_value
    else:
        magnitude_spectrum = s
    # 通过对数转换使得幅度谱更易于观察
    # log_magnitude_spectrum = 20 * np.log10(magnitude_spectrum)
    log_magnitude_spectrum = magnitude_spectrum

    # 设置频谱图的坐标轴范围
    frequency_bins = np.arange(s.shape[0])
    time_frames = np.arange(s.shape[1])

    # 绘制频谱图
    plt.imshow(log_magnitude_spectrum, aspect='auto', origin='lower', cmap='jet',
               extent=[time_frames[0], time_frames[-1], frequency_bins[0], frequency_bins[-1]])

    # 设置坐标轴标签
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')

    # 显示颜色条
    plt.colorbar(label='Magnitude (dB)')

    # name = file_path.split('.')[0]
    # 显示频谱图
    plt.savefig(f"./specturm/{k}_{flag}.png")
    plt.close()


for k in range(len(fake)):
    a_fake = fake[k]
    a_real = real[k]

    fake_re = np.zeros((101, 2001))
    real_re = np.zeros((101, 2001))
    for i in range(101):
        for j in range(2001):
            fake_re[i, j] = np.abs(a_fake[0, i, j] + 1j * a_fake[1, i, j])
            real_re[i, j] = np.abs(a_real[0, i, j] + 1j * a_real[1, i, j])
    draw(fake_re, 'fake', k)
    draw(real_re, 'real', k)
