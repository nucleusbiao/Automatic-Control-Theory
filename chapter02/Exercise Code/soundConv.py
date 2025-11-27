import numpy as np
from scipy import signal
import soundfile as sf
import sounddevice as sd

# 导入声音
IR, fs_ir = sf.read('IR.wav')
talk, fs_talk = sf.read('talk.wav')

# 确保采样率一致
if fs_ir != fs_talk:
    print("警告：两个音频文件的采样率不同！")
    # 这里可以添加重采样代码如果需要

# 卷积
mix = np.zeros((len(talk) + len(IR) - 1, 2))
mix[:, 0] = signal.convolve(talk[:, 0], IR[:, 0])
mix[:, 1] = signal.convolve(talk[:, 1], IR[:, 1])

# 播放
sd.play(mix, fs_ir)  # 使用IR的采样率，或者使用fs_talk
sd.wait()  # 等待播放完成

# 可选：保存结果文件
# sf.write('mix.wav', mix, fs_ir)