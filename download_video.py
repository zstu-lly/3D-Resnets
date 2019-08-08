import random
from threading import Thread
import cv2

import logging
from utils import *
from queue import Queue
import os
import shutil
import numpy as np


def get_finished():
    finished = []
    with open("finish.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            finished.append(line.strip('\n'))
    return finished


logging.basicConfig(level=logging.DEBUG,
                    filemode='w',    # 'w' or 'a'
                    filename='output.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(threadName)s - %(levelname)s - %(lineno)d - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)


class Downloader:

    def __init__(self, mode, ids, clip_length):
        self.mode = mode
        self.ids = ids    # a list of video filename
        self.clip_length = clip_length
        self.delete_queue = Queue()
        self.task_queue = Queue()
        self.build_data()
        self.finished = set(get_finished())

    def build_data(self):
        for video_filename in self.ids:
            self.task_queue.put(video_filename)

    def extra_frames(self, video_filename):
        # if "webm" in video_filename:
        #     return

        frames_root_dir = os.path.join(self.mode, "frames")  # 根目录
        frames_dir = os.path.join(frames_root_dir, video_filename.split('.')[0])
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)
        current_frame_count = len(os.listdir(frames_dir))
        dest_frame_count = self.clip_length * 10
        if current_frame_count == dest_frame_count:  # 如果已经提取过了图片,直接跳过
            self.check_size(frames_dir)
            print("目前帧数%d,跳过%s" % (current_frame_count, video_filename))
            with open("finish.txt", 'a') as f:
                f.write(video_filename + '\n')
            return

        self.download_video(video_filename)
        video_path = os.path.join(self.mode, "video", video_filename)  # 视频文件的路径
        cap = cv2.VideoCapture(video_path)
        available_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"{video_filename}, 总帧数:{available_frame_count}, 帧率:{fps}")    # 获取所有帧数

        if available_frame_count < dest_frame_count:
            logger.warning("%s可提取的帧只有%d" % (video_filename, available_frame_count))
            dest_frame_count = available_frame_count

        print("开始提取%s的视频帧,可用%d帧,准备提取%d帧" % (video_filename, available_frame_count, dest_frame_count))
        if available_frame_count > 20000:
            return

        # 先把文件夹里面的图片都删除掉
        if current_frame_count == dest_frame_count:  # 如果已经提取过了图片,直接跳过
            print("目前帧数%d,跳过%s" % (current_frame_count, video_filename))
            # 应该先对里面的图片进行缩放
            self.check_size(frames_dir)
            with open("finish.txt", 'a') as f:
                f.write(video_filename + '\n')
            return
        else:    # 如果没提取完
            print(f"先对{frames_dir}进行了清除, 帧数只有{current_frame_count}")
            for file in os.listdir(frames_dir):
                path = os.path.join(frames_dir, file)
                os.remove(path)

        start_time = time.time()
        sample_frequency = available_frame_count//dest_frame_count
        start_index = random.randint(0, sample_frequency)
        count = 0
        for i in range(int(dest_frame_count)):
            index = int(start_index+i*sample_frequency)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, frame = cap.read()
            if success:
                frame_path = os.path.join(frames_dir, str(index).zfill(4) + '.jpg')
                count += 1
                # 对图片进行缩放
                frame = self.resize(frame)
                cv2.imwrite(frame_path, frame)

        cap.release()
        logger.info(f"提取%s的视频帧完成,可用%d帧,一共提取了%d帧, 耗时:{time.time() - start_time}" % (video_filename, available_frame_count, count))
        with open("finish.txt", 'a') as f:
            f.write(video_filename + '\n')

    def check_size(self, frames_dir):
        for file in os.listdir(frames_dir):
            path = os.path.join(frames_dir, file)
            frame = cv2.imread(path)
            result = self.resize(frame)
            if result is not None:
                logger.debug(f"对{path}重新缩放")
                # os.remove(path)
                cv2.imwrite(path, result)
            # else:

    def resize(self, frame):
        # 缩放大最大边为256
        max_size = 256
        # frame = np.copy(frame)
        h, w, c = frame.shape
        if max(h, w) == max_size:
            return None

        if h > w:
            scale = h / max_size
            frame = cv2.resize(frame, (int(w/scale), max_size))
        else:
            scale = w / max_size
            frame = cv2.resize(frame, (max_size, int(h/scale)))
        # print(h, w, c, frame.shape)
        return frame

    """
    def get_available_frame_count(self, video_filename):
        video_path = os.path.join(self.mode, "video", video_filename)  # 视频文件的路径
        if not os.path.exists(video_path):
            self.download_video(video_filename)
        cap = cv2.VideoCapture(video_path)
        count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                count += 1
            else:
                break
        cap.release()
        print("%s的可提取的帧计算完成, 一共%d帧" % (video_filename, count))
        return count
    """

    def download_video(self, video_name):
        root = 'http://media-videonet.videojj.com/'
        videos_dir = os.path.join(self.mode, "video")
        if not os.path.exists(videos_dir):
            os.makedirs(videos_dir)
        url = root + video_name
        os.system('wget -c 0 -P %s %s -q' % (videos_dir, url))  # -c: 断点续传 -P: 进度条    自动去重
        if not os.path.exists(os.path.join(videos_dir, video_name)):    # 如果下载失败的话
            logger.error("视频%s下载失败" % video_name)
        else:
            print("视频%s下载成功" % video_name)

    def delete_file(self):
        while True:
            path = self.delete_queue.get()
            if not os.path.exists(path):
                logger.debug("%s路径不存在" % path)
                continue
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)  # 删除帧文件夹以及这个文件夹下面所有的帧
                else:
                    os.remove(path)
            except Exception as e:
                logger.error(e)

    def download(self):
        while True:
            try:
                video_filename = self.task_queue.get()
                print("进度:", self.ids.index(video_filename), '-', len(self.ids))
                self.extra_frames(video_filename)
                video_file_path = os.path.join(self.mode, "video", video_filename)
                if os.path.exists(video_file_path):
                    self.delete_queue.put(video_file_path)
            except Exception as e:
                logger.error(e)

    def run(self):
        ths = []
        thread = Thread(target=self.delete_file)
        thread.start()
        ths.append(thread)

        for i in range(3):
            thread = Thread(target=self.download)
            thread.start()
            ths.append(thread)

        for th in ths:
            th.join()


if __name__ == '__main__':
    partition, labels = get_partition_labels()
    mode = "train"
    train_file_dir = f'{mode}'
    train_generator = Downloader(mode=train_file_dir, ids=partition[mode], clip_length=clip_length)
    train_generator.run()


