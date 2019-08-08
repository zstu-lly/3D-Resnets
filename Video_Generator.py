import random
import torchvision
import keras
import numpy as np
from PIL import Image
import cv2
import logging
from utils import *
from queue import Queue
import os
from config import *
from transforms import *


class Stack:

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=0)
        elif img_group[0].mode == 'RGB':
            return np.stack([np.array(x)[:, :, ::-1] for x in img_group], axis=0)


logging.basicConfig(level=logging.DEBUG,
                    filemode='w',  # 'w' or 'a'
                    filename='generator.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(threadName)s - %(levelname)s - %(lineno)d - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)


class DataGenerator(keras.utils.Sequence):

    def __init__(self, mode, ids, labels, classes, img_size, clip_length=8, batch_size=8,
                 shuffle=False):
        self.transform = torchvision.transforms.Compose(
            [GroupMultiScaleCrop(img_size, [1, .875, .75, .66]),
             GroupRandomHorizontalFlip(is_flow=False),
             Stack()])

        self.mode = mode
        self.ids = ids  # a list of video filename
        self.labels = labels  # a dict map from filename to class
        self.classes = classes  # a list of all classes
        self.num_classes = len(classes)    # 353
        self.clip_length = clip_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.ids))
        self.delete_queue = Queue()
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.ids) // self.batch_size  # 不要最末尾的那几个

    def __getitem__(self, index):
        t1 = time.time()
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_ids = [self.ids[i] for i in batch_indexes]
        X, Y = self.data_generation(batch_ids)
        t2 = time.time()
        logger.info(str(batch_indexes) + ' ' + str(t2 - t1))
        return X, Y

    def data_generation(self, ids):
        X_data = np.zeros(shape=(batch_size, clip_length, img_size, img_size, 3))
        Y_data = []
        for index, id in enumerate(ids):
            frames, label = self.next_sample(id)  # 获取下一个样本
            X_data[index] = frames
            Y_data.append(label)
        X_data -= 128.0
        X_data /= 128.0
        Y_data = keras.utils.to_categorical(Y_data, num_classes=self.num_classes)
        return X_data, Y_data

    def next_sample(self, video_name):
        try:
            sampled_frames = self.get_frames(video_name)  # 对这个视频进行帧采样
            video_class = self.labels[video_name]
            video_class_index = self.classes.index(video_class)  # 获取这个视频类别的下标
            # logger.debug(video_name + ' ' + video_class + ' ' + str(video_class_index))
            return sampled_frames, video_class_index  # 返回采样得到的16帧和视频类别的下标
        except Exception as e:
            logger.error(f"{video_name, e}")
            return self.next_sample(random.choice(self.ids))  # 随便换个视频

    # 帧采样算法
    def get_frames(self, video_name):
        try:
            # 在已经提取了视频帧的前提下
            frames_root_dir = os.path.join(self.mode, "frames")  # 根目录
            frames_dir = os.path.join(frames_root_dir, video_name.split('.')[0])
            # 进行采样
            filenames = os.listdir(frames_dir)
            filenames = sorted(filenames, key=lambda x: int(x.split('.')[0]))
            # filenames = sorted(filenames)
            sampling_frequency = len(filenames) // self.clip_length  # 采样频率
            # sampling_frequency = 1
            # logger.debug("%s的采样频率:%d" % (video_name, sampling_frequency))
            count = 0
            start_index = random.randint(0, len(filenames) // self.clip_length-1)  # 起始下标    # TODO error
            # start_index = random.randint(0, len(filenames)-self.clip_length)  # 起始下标
            file_index = start_index
            clip = []
            while True:
                image_path = os.path.join(frames_dir, filenames[file_index])
                # logger.debug(image_path)
                img = Image.open(image_path).convert('RGB')
                clip.append(img)
                count += 1
                if count == self.clip_length:
                    break
                file_index += sampling_frequency
            # t1 = time.time()
            process_data = self.transform(clip)
            # logger.info(f"{video_name}, 数据增强耗时:{time.time()-t1}")
            return process_data
        except Exception as e:
            raise e


if __name__ == '__main__':
    partition, labels = get_partition_labels()
    classes = get_all_classes('classes.txt')

    train_generator = DataGenerator(classes=classes, mode=train_file_dir, ids=partition['train'],
                                    labels=labels, batch_size=batch_size, clip_length=clip_length, img_size=img_size)
    for i in train_generator:
        pass

    # val_generator = DataGenerator(classes=classes, mode=val_file_dir, ids=partition['val'],
    #                               labels=labels, batch_size=batch_size, clip_length=clip_length)
    # for i in val_generator:
    #     pass
