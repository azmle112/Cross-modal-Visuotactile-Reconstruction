"""STFT"""

import os
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt


def read_text_files(folder_path):
    text_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            text_files.append(os.path.join(folder_path, file))
    return text_files


# 定义生成频谱图的函数
def generate_spectrogram(file_path, index):
    # 读取文本文件中的数据
    data = np.loadtxt(file_path)
    if len(data) > 200000:
        data = data[:200000]
    else:
        data = np.pad(data[:200000], (0, 200000 - len(data)), mode='constant')
    # 计算STFT

    fs = 1000  # 采样频率
    window_length = 200  # 窗宽为50
    f, t, Z = stft(data, fs=fs, nperseg=window_length, window='hann')

    # 分离实部和虚部
    real_part = np.real(Z)
    imag_part = np.imag(Z)

    # 重构触觉频谱S
    S = np.concatenate((real_part[np.newaxis, :, :], imag_part[np.newaxis, :, :]), axis=0)

    # 打印触觉频谱S的形状
    spectrogram_shape = S.shape
    print("Spectrogram shape:", spectrogram_shape)

    max_value = np.max(S)
    min_value = np.min(S)

    # 将s归一化到[-1, 1]的范围
    normalized_s = 2 * (S - min_value) / (max_value - min_value) - 1
    # S的形状为(2, frequency_bins, time_frames)，其中frequency_bins表示频率点的数量，time_frames表示时间帧的数量

    return normalized_s


# 指定文件夹路径
directory_path = '___LMT_TextureDB___/Training/Accel/'
test_path = '___LMT_TextureDB___/Testing/AccelScansDFT321'
# 读取所有文本文件
text_files = read_text_files(directory_path)
# text_files = read_text_files(test_path)
# 生成频谱图
spec_list = []
i = 0
for file_path in text_files:
    s = generate_spectrogram(file_path, i)
    spec_list.append(s)
    print(i)
    i += 1
# spec_np = np.stack(np.array(spec_list))
# np.save("./spec2.npy", spec_np)
