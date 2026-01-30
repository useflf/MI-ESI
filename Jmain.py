import scipy.io as sio
import numpy as np
from J.REML import reml
from J.l_curve import l_curve


# 切割为子图qi
def cut(input_list):
    result_lists = []
    list_length = len(input_list[0])
    i = 0

    while i + 14 <= list_length:
        result_lists.append([row[i: i + 14] for row in input_list])
        i += 14

    return result_lists


# 切割EEG信号为多个窗口
def split_eeg_signal(eeg_signal, window_number, overlap_ratio=0.5):
    signal_length = eeg_signal.shape[1]
    window_size = int(2 * signal_length // (window_number + 1))
    overlap_size = int(window_size * overlap_ratio)

    windows = []
    start = 0
    end = window_size

    for _ in range(window_number):
        window = eeg_signal[:, start:end]
        windows.append(window)
        start = end - overlap_size
        end = start + window_size

    return windows


# 计算J值
def J(inter_file, eeg_path):
    # 加载数据
    load = np.array(inter_file['interpolate_res'])
    data_index = load[:, :, 4][0]
    G_data = np.load('./file/lead_field_matrix.npy')
    data_eeg = sio.loadmat(eeg_path)['x']

    different_indices = (np.where(data_index[:-1] != data_index[1:])[0] + 1).tolist()
    index = [0] + different_indices + [len(data_index)]

    data = load[:, :, 3]

    R = []
    G_list = []
    for i in range(len(index) - 1):
        Qi = []
        qi = []
        G = []

        qi += cut([row[index[i]: index[i+1]] for row in data])
        qi_array = np.array(qi)

        # G降维以匹配维度
        G += cut(([row[index[i]: index[i+1]] for row in G_data]))  # 丢失信息不准确
        g = np.array(G)
        G_array = np.zeros_like(g[0])
        for n in range(len(G)):
            G_array += g[n]
        G_array /= len(G)
        G_list.append(G_array)

        # 计算Qi
        for j in range(len(qi)):
            Qi.append(np.dot(qi_array[j].T, qi_array[j]))
        Qi_array = np.array(Qi)

        # 分割窗口以计算lamdaR
        lamda = []
        window_number = data[:, index[i]:index[i+1]].shape[1] // 14
        windows = split_eeg_signal(data_eeg, window_number)
        for window_signal in windows:
            lamda.append(reml(G_data, window_signal))

        # 计算R
        R_array = np.zeros_like(Qi[0])
        for k in range(Qi_array.shape[0]):
            R_array += lamda[k] * Qi_array[k]
        R_array /= Qi_array.shape[0]
        R.append(R_array)
        if i + 1 % 10 == 0:
            print('Q: ' + str(i + 1) + '/' + str(len(index) - 1))

    # 计算J
    _J = []
    for i in range(len(R)):
        inverse_matrix = np.linalg.inv((np.dot(np.dot(G_list[i], R[i]), G_list[i].T)
                                                   + l_curve(R[i], G_list[i], data_eeg) * np.identity(len(data_eeg))))
        j = np.dot(np.dot(np.dot(R[i], G_list[i].T), inverse_matrix), data_eeg)
        _J.append(np.mean(j, axis=0))

        if (i + 1) % 10 == 0:
            print('J: ' + str(i + 1) + '/' + str(len(R)))

    return {'label': inter_file['label'], 'J': np.array(_J)}
