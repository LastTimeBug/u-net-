import os
import glob
from PIL import Image
import subprocess
import re
# 定义文件夹路径
import numpy as np


def getdata(folder_path):
    # 获取文件夹中的所有图像文件路径
    image_files = glob.glob(os.path.join(folder_path, "*.jpg"))  # 假设您的帧是.jpg格式的图像文件
    # 创建一个列表来存储帧
    frames = []
    # 遍历图像文件路径并将每个帧添加到列表中
    for image_file in image_files:
        # 读取图像文件并将帧添加到列表
        image = Image.open(image_file)  # 自定义函数，根据实际情况读取图像文件
        frames.append(image)
    # 打印帧的数量
    print(len(frames))
    return  frames



def load_images_and_labels(directory, labels):
    # 获取目录中的所有图像文件
    image_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.jpg')]
    # 确保图像文件数量和标签数量一致
    if len(image_files) != len(labels):
        raise ValueError("图像文件数量与标签数量不一致")

    # 创建一个空的NumPy数组来存储所有图像数据
    dataset = np.empty((len(image_files), 720, 1280, 3))

    # 逐个加载和处理图像文件
    for i, file_path in enumerate(image_files):
        img = Image.open(file_path)

        # 调整图像尺寸为1280x720
        img = img.resize((1280, 720))

        # 将图像数据转换为NumPy数组
        img_array = np.array(img)

        # 将图像数据存储到数据集中
        dataset[i] = img_array

    return dataset, labels


def load_images_as_array(directory):
    # 获取目录中的所有图像文件
    image_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.jpg')]

    # 创建一个空的NumPy数组来存储所有图像数据
    dataset = np.empty((len(image_files), 720, 1280, 3))

    # 逐个加载和处理图像文件
    for i, file_path in enumerate(image_files):
        img = Image.open(file_path)

        # 调整图像尺寸为1280x720
        img = img.resize((1280, 720))

        # 将图像数据转换为NumPy数组
        img_array = np.array(img)

        # 将图像数据存储到数据集中
        dataset[i] = img_array

    return dataset
def get_keyframe_timestamps(video_file):
    # 使用FFprobe命令行工具获取视频关键帧的时间戳
    command = f'ffprobe -select_streams v -show_frames -show_entries frame=pkt_pts_time -of csv=p=0 {video_file}'
    output = subprocess.check_output(command, shell=True).decode('utf-8')
    # 解析FFprobe输出，提取关键帧时间戳
    timestamps = re.findall(r'[0-9]+\.[0-9]+', output)
    # 将时间戳转换为浮点数
    timestamps = [float(timestamp) for timestamp in timestamps]
    # 获取关键帧的起始时间和终止时间
    start_time = timestamps[0]
    end_time = timestamps[-1]
    return start_time, end_time


def cut_video(input_file, output_file, start_time, end_time):
    # 使用FFmpeg命令行工具切割视频
    command = f'ffmpeg -i {input_file} -ss {start_time} -to {end_time} -c copy {output_file}'
    subprocess.call(command, shell=True)
