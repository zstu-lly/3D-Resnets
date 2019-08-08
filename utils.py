
from pynvml import *
import os
from config import clip_length, train_file_dir, val_file_dir
import time


def get_all_classes(path):
    lst = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            lst.append(line)
    print(f"class数量:{len(lst)}")
    return lst


def get_partition_labels():
    """

    :return: 返回{'train': [...], 'val': [...]}
    """
    partition = dict()
    partition['train'] = list()
    partition['val'] = list()
    labels = dict()
    print("开始加载所有的视频文件名和及其类别")
    with open(os.path.join(train_file_dir, "list.txt"), 'r') as f:
        for line in f:
            line = line.strip('\n')
            video_name, label = line.split(' ')    # 视频名以及类别名
            partition['train'].append(video_name)
            labels[video_name] = label
    with open(os.path.join(val_file_dir, "list.txt"), 'r') as f:
        for line in f:
            line = line.strip('\n')
            video_name, label = line.split(' ')    # 视频名以及类别名
            partition['val'].append(video_name)
            labels[video_name] = label
    print("加载完成")
    print(f"train:{len(partition['train'])}, val:{len(partition['val'])}")
    return partition, labels


def set_visible_devices():
    env, count = get_available_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = env
    return count


def get_available_gpu():
    """

    :return: 可以用的gpu的标号
    """
    available_gpu_ids = []
    count = 0
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(i, 'Total:', info.total, 'Free:', info.free, 'Used:', info.used)
        if info.free > 10000000000:
            available_gpu_ids.append(str(i))
            count += 1
    nvmlShutdown()
    print("共有%d个gpu可用:" % count, available_gpu_ids)
    if count == 0:
        time.sleep(30)
        return get_available_gpu()
    else:
        # return ','.join(available_gpu_ids[]), count
        return ','.join(available_gpu_ids[:1]), 1


if __name__ == '__main__':
    get_available_gpu()
