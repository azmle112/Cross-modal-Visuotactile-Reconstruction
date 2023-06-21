# Cross-modal-Visuotactile-Reconstruction

该项目用于对加速度数据进行STFT变换，并对相应物体图片进行特征提取、绘制频谱图以及利用GAN网络进行频谱的重构。

## 文件说明

- `stft2.py`: 对加速度数据进行STFT（短时傅里叶变换）处理的Python脚本。
  - 输入：加速度数据文本
  - 输出：频谱数据

- `extract.py`: 对原始图像进行特征提取的Python脚本。
  - 输入：原始图像
  - 输出：提取的特征数据

- `draw_spec.py`: 绘制频谱图的Python脚本。
  - 输入：频谱数据
  - 输出：频谱图像

- `reconstruct.py`: 利用GAN（生成对抗网络）重构频谱的Python脚本。
  - 输入：频谱数据
  - 输出：重构后的频谱数据



