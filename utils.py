import cv2
import torch
from torch import Tensor
from torch.distributions import Categorical
import collections
import numpy as np
from typing import Union
import shutil
import stat
import os
from PIL import Image
from moviepy.editor import *
import random
import glob
from imageR import img2emotion
from params import Params
from keras.models import load_model

def get_img_emo_index():
    param = Params()
    model = load_model(os.path.join(param.img_model_path, "model_filter.h5"))
    folder_path = "./Repo/video/sub_mp4/"  # 分割后视频的路径
    dir_path = './Repo/data/img/'  # 图片的路径
    file_names = []
    img_numbers = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_name_without_extension = os.path.splitext(file)[0]  # 去除后缀的文件名
            file_names.append(file_name_without_extension)
    for number in file_names:
        file_prefix = number  # 以不同的时间分割区间去区分图片
        count_dict = [0, 0, 0]
        for root, dirs, files in os.walk(dir_path):
            for file in files:  # 找到图片中所有以file_prefix开头的图片
                if file_prefix in file:
                    img_path = os.path.join(dir_path, file)
                    (img_emotion, img_index) = img2emotion(img_path, param, model)
                    if img_index is not None:
                        count_dict[img_index] += 1
                    # 在这里对满足条件的文件进行操作

        max_count = max(enumerate(count_dict), key=lambda x: x[1])[0]  # 统计这段分割视频中标签的数量，找出最大值
        img_numbers.append(max_count)

    return img_numbers


def video2data(path, cuts=[], p=2):
    """
    path: mp4文件路径
    cut = [(begin, end), (begin, end), ...]: 视频分段剪切
    p: 定义每秒抽取的图片数目
    """
    _, mp4_name = os.path.split(path)
    mp4_name = mp4_name.split(".")[0]
    mp4 = VideoFileClip(path)
    sub_mp4 = list()
    if not len(cuts):
        sub_mp4.append(mp4)
    for sub_clip in cuts:
        sub_mp4 = mp4.subclip(sub_clip[0], sub_clip[1])
        sub_mp4.write_videofile(os.path.join("./Repo/video/sub_mp4/", mp4_name + f"sub{sub_clip[0]}_{sub_clip[1]}.mp4"))
    mp4.close()

    # 切完,开抽
    img_path = "./Repo/data/img"
    audio_path = "./Repo/data/audio"

    # 音频抽取
    print("抽音...")
    for sub_mp4_name in os.listdir("./Repo/video/sub_mp4"):
        this_mp4_path = os.path.join("./Repo/video/sub_mp4/", sub_mp4_name)
        sub_mp4 = AudioFileClip(this_mp4_path)
        sub_mp4.write_audiofile(os.path.join("./Repo/data/audio/", sub_mp4_name.split(".")[0] + ".wav"))
        sub_mp4.close()
    print("抽音完成")

    # 图片抽取
    print(mp4_name, "抽帧...")
    idx = 0
    for sub_mp4 in os.listdir("./Repo/video/sub_mp4"):
        idx += 1
        c = 1
        save_tag = 1
        this_mp4_path = os.path.join("./Repo/video/sub_mp4/", sub_mp4)
        this_capture = cv2.VideoCapture(this_mp4_path)
        fps = this_capture.get(5)
        while True:
            ret, frame = this_capture.read()
            if ret:
                frame_rate = int(fps) // p
                if c % frame_rate == 0:
                    # 抽取
                    cv2.imwrite(os.path.join(r"./Repo/data/img/", sub_mp4.split(".")[0] + f"{idx}_{save_tag}.jpg"), frame)
                    save_tag += 1
                c += 1
                cv2.waitKey(0)
            else:
                print(sub_mp4, "抽帧完成")
                break


def img2gif(p):
    gif_dict = dict()
    for img_name in os.listdir(p.img_results_path):
        keys = img_name.split("_")
        img_path = os.path.join(p.img_results_path, img_name)
        img = cv2.imread(img_path)
        w = img.shape[1]
        h = img.shape[0]
        if str(keys[0]) + str(keys[1]) in gif_dict.keys():
            gif_dict[str(keys[0]) + str(keys[1])].append((int(keys[2].split(".")[0]), img))
        else:
            gif_dict[str(keys[0]) + str(keys[1])] = [(int(keys[2].split(".")[0]), img)]
    
    for gif_key in gif_dict.keys():

        gif = cv2.VideoWriter(os.path.join(p.gif_results_path, str(gif_key) + ".avi"), cv2.VideoWriter_fourcc("D","I","V", "X"), 2, (w ,h))

        img_list = gif_dict[gif_key].copy()
        img_list.sort(key=lambda x: x[0])

        for img in [item[1] for item in img_list]:
            gif.write(img)
        
        gif.release()
        cv2.destroyAllWindows()

def get_video_duration(video_path):
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    # 获取视频的总帧数和帧率
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 计算视频的时间长度
    duration = frame_count / fps
    # 关闭视频
    cap.release()
    return duration


def generate_random_integers(limit:int, num_integers:int):
    interval_length = limit / num_integers  # 确定每个区间的长度
    start_time = 0.0
    integers = []
    for _ in range(num_integers):
        end_time = start_time + interval_length  # 计算当前区间的结束时间
        integer = (int(start_time), int(end_time))  # 构建区间元组
        integers.append(integer)  # 将区间添加到列表中
        start_time = end_time  # 下一个区间的起始时间为当前区间的结束时间
    return integers


def compute_prob(x:list):
    counts = collections.Counter(x)
    total = len(x)
    probs = []
    for k, v in counts.items():
        probs.append(v / total)
    probs = np.array(list(counts.values())) / total
    return torch.tensor(probs)

def compute_mi(x:Union[Tensor, list], y:Union[Tensor, list]):
    if isinstance(x, Tensor):
        x = x.detach().tolist()
    if isinstance(y, Tensor):
        y = y.detach().tolist()
    x_ten = compute_prob(x)
    y_ten = compute_prob(y)

    px = Categorical(probs=x_ten)
    py = Categorical(probs=y_ten)
    pxy = Categorical(probs=torch.outer(x_ten, y_ten))

    mi = torch.sum(pxy.log_prob(pxy.sample()) - px.log_prob(px.sample()) - py.log_prob(py.sample()))
    return mi.item()


def prepare_video(src_path, tar_path):
    """
    把原视频移动到目标路径下
    """
    shutil.copy(src_path, tar_path)

def clear_file(para):
    delete_all_files_and_folders(para.img_path)
    delete_all_files_and_folders(para.audio_path)
    delete_all_files_and_folders(para.mp4_path)
    delete_all_files_and_folders(para.sub_mp4_path)

def delete_all_files_and_folders(folder):
    for subfolder in os.listdir(folder):
        full_path = os.path.join(folder, subfolder)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        os.chmod(file_path, stat.S_IWRITE)
        os.unlink(file_path)
