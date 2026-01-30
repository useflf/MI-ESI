from interpolate.location import get_locat
import scipy.io as sio
import copy
from interpolate.interpolate import get_interpolation

# 插值函数
def interpolate(res_file):
    # 加载通道位置信息
    path_locat = './file/projection.xlsx'
    axis = 'opt coord(MNI)'
    location = get_locat(path_locat, axis)

    # 获取脑信号数据
    fnirs_deoxy = res_file['x']
    lst = []

    exclude = []
    for time in range(len(fnirs_deoxy[1])):
        lst_per = []
        for chan in range(len(location)):
            if chan in exclude:
                continue
            temp = copy.deepcopy(location[chan])
            temp.append(fnirs_deoxy[chan][time])
            lst_per.append(temp)
        lst.append(lst_per)

    # 加载插值点的位置信息
    path_point = './file/point_loc.mat'
    all_points = sio.loadmat(path_point)['downsampled'][:, :]
    res = []
    length = len(fnirs_deoxy[1])

    # 循环在每个时间点和通道上插值
    for time in range(length):
        res_per = []
        for x, y, z, label in all_points:
            value_eval = get_interpolation(x, y, z, lst[time])
            res_per.append([x, y, z, value_eval, label])
        res.append(res_per)

        if (time + 1) % 10 == 0:
            print(str(time + 1) + ' / ' + str(length))

    data_save = {'label': res_file['label'], 'interpolate_res': res}

    return data_save
