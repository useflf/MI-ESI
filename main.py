import scipy.io as sio
from interpolatemain import interpolate
from Jmain import J
from WPLI.WPLI import calculate
import os
import numpy as np
from filter import preprocess


root_folder = './EEG-NIRS/NIRS'

dic = {'7': 0, '8': 2, '9': 3}
for root, dirs, files in os.walk(root_folder):
    # 遍历当前文件夹下的所有子文件夹
    for subfolder in dirs:
        # 切割epoch，并搭上标签并保存
        eeg_path = os.path.join(root_folder[: -4] + 'EEG', subfolder[: -4] + 'EEG')
        nir_cnt_path = os.path.join(root, subfolder) + '/cnt_nback.mat'
        eeg_cnt_path = os.path.join(eeg_path, 'cnt_nback.mat')
        nir_mrk_path = os.path.join(root, subfolder) + '/mrk_nback.mat'
        eeg_mrk_path = os.path.join(eeg_path, 'mrk_nback.mat')

        nir_cnt_data = np.swapaxes(sio.loadmat(nir_cnt_path)['cnt_nback'][0][0][0][0][0]['x'], 0, 1)
        eeg_cnt_data = np.swapaxes(sio.loadmat(eeg_cnt_path)['cnt_nback'][0][0]['x'], 0, 1)
        nir_mrk_data = sio.loadmat(nir_mrk_path)
        eeg_mrk_data = sio.loadmat(eeg_mrk_path)
        nir_time = np.floor_divide(nir_mrk_data['mrk_nback']['time'][0][0][0], 100)
        eeg_time = np.floor_divide(eeg_mrk_data['mrk_nback']['time'][0][0][0], 5)
        label = nir_mrk_data['mrk_nback']['event'][0][0]['desc'][0][0]

        index = 0
        for p in range(len(label)):
            index += 1
            file_name = str(dic[str(label[p][0])]) + '-' + str(index) + '-' + subfolder
            nir_data_save_path = os.path.join('./file/data/NIRS/', file_name) + '.mat'
            eeg_data_save_path = os.path.join('./file/data/EEG/', (file_name[:11] + 'EEG')) + '.mat'

            # 分割epoch
            nir_baseline = nir_cnt_data[:, nir_time[p]-50: nir_time[p]-20]
            eeg_baseline = eeg_cnt_data[:, eeg_time[p*21+1] - 1000: eeg_time[p*21+1] - 400]

            # 滤波并保存
            nir_res = {'label': dic[str(label[p][0])], 'x': preprocess(nir_cnt_data[:, nir_time[p]+20: nir_time[p]+420],
                                                                       nir_baseline, _type='fnirs')}
            eeg_res = {'label': dic[str(label[p][0])], 'x': preprocess(eeg_cnt_data[:, eeg_time[p * 21 + 1]:
                                                        eeg_time[(p + 1) * 21 - 1] + 400], eeg_baseline, _type='eeg')}
            sio.savemat(nir_data_save_path, nir_res)
            sio.savemat(eeg_data_save_path, eeg_res)

            # 投影插值并保存
            inter_res_file = interpolate(nir_res)
            inter_save_path = os.path.join('./file/interpolate', file_name) + '.mat'
            sio.savemat(inter_save_path, inter_res_file)

            # 重建电流密度并保存
            J_res_file = J(inter_res_file, eeg_data_save_path)
            J_save_path = os.path.join('./file/J', file_name) + '.mat'
            sio.savemat(J_save_path, J_res_file)

            # 计算WPLI并保存
            WPLI_res_file = calculate(J_res_file)
            WPLI_save_path = os.path.join('./file/WPLI', file_name) + '.mat'
            sio.savemat(WPLI_save_path, WPLI_res_file)
