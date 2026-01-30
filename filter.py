import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import mne
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def ocular_artifact(eeg_data, fs):
    # 定义脑电通道名称
    ch = ['Fp1', 'Fp2', 'AFF5h', 'AFF6h', 'AFz', 'F1', 'F2', 'FC1', 'FC2', 'FC5', 'FC6', 'Cz', 'C3', 'C4', 'T7', 'T8',
          'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'P7', 'P8', 'POz', 'O1', 'O2', 'TP9', 'TP10']
    print(len(eeg_data[0]))
    # 创建 MNE 信息对象
    info = mne.create_info(ch_names=ch, sfreq=fs, ch_types='eeg')
    # 使用 MNE 创建原始数据对象
    raw = mne.io.RawArray(eeg_data, info)
    # 初始化独立成分分析对象
    ica = ICA(n_components=15, random_state=97)
    # 对原始数据应用独立成分分析
    ica.fit(raw)

    # 排除识别为眼动伪迹的成分
    ica.exclude = [0, 1, 2]
    # 应用独立成分分析处理后的数据
    ica.apply(raw)
    # 获取处理后的数据
    res = raw.get_data()

    return res


def eeg_filter(eeg_signal, eeg_fs, low_cut, high_cut):
    # 使用 ocular_artifact 函数处理脑电信号
    pre = ocular_artifact(eeg_signal, eeg_fs)
    nyquist = 0.5 * eeg_fs
    low = low_cut / nyquist
    high = high_cut / nyquist
    # 设计巴特沃斯带通滤波器
    b, a = signal.butter(6, [low, high], btype='band')

    return signal.filtfilt(b, a, pre)


def fnirs_filter(nir_signal, fnirs_fs, cutoff_freq, order=2):
    nyquist = 0.5 * fnirs_fs
    normal_cutoff = cutoff_freq / nyquist
    # 设计巴特沃斯低通滤波器
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, nir_signal)


def preprocess(_signal, baseline, _type):
    if _type == 'eeg':
        # 预处理脑电信号
        res = eeg_filter(_signal, 200, 1, 40) - np.tile(np.mean(baseline, axis=1), (_signal.shape[1], 1)).T
        return res

    elif _type == 'fnirs':
        # 预处理脑血氧信号
        res = fnirs_filter(_signal, 10, 0.2) - np.tile(np.mean(baseline, axis=1), (_signal.shape[1], 1)).T
        return res

    else:
        print("Baseline correction error\n")


def visualize_filter(_signal, start_index, end_index):
    # 提取信号片段
    segment = _signal[:, start_index:end_index]
    # 创建时间轴
    time_axis = np.arange(start_index, end_index)

    # 绘制时域图
    plt.figure(figsize=(10, 6))
    for i in range(segment.shape[0]):
        plt.plot(time_axis, segment[i], label=f'Channel {i + 1}')

    plt.title('Signal Time Domain Plot')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
