import myutils
import torch.nn as nn
import torch
from PIL import Image
from torchvision.transforms import ToTensor, Resize
from torchvision import transforms
from unetmodel import  UNet
import numpy as np
import torch
def preprocess_data(data):
    transformed_data = []
    transform = transforms.Compose([
        Resize((360,640 )),
        ToTensor()
    ])

    for image in data:
        tensor_image = transform(image)
        # 将图像的张量形式添加到列表中
        transformed_data.append(tensor_image)
    return torch.stack(transformed_data)
# 加载模型
model = UNet()
model.load_state_dict(torch.load('D:/data/model/u-netModel.pth'))
model.eval()  # 将模型设置为评估模式
folder_path = r'D:\data\test3'
data = myutils.getdata(folder_path)
print(type(data))
data = preprocess_data(data)
with torch.no_grad():
    outputs = model(data)
predicted_mask = outputs.squeeze().numpy()  # 将批次维度去除，并转换为 numpy 数组
# 对预测结果进行二值化处理，这里假设阈值为0.5
predicted_mask = np.where(predicted_mask > 0.5, 1, 0)
print(predicted_mask)