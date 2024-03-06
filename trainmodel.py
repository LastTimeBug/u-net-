import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import myutils
import torch
from torch.utils.data import Dataset
import numpy as np
# 创建卷积神经网络模型
from unetmodel import UNet
from torchvision.transforms import Resize, ToTensor
from torchvision import transforms
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y


def train_model(folder_path, train_labels, model):
    # 创建自定义数据集
    train_data = myutils.getdata(folder_path)
    transform = transforms.Compose([transforms.Resize((360, 640)), transforms.ToTensor()])
    train_dataset = MyDataset(train_data, train_labels, transform=transform)  # 自定义数据集类
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 进行训练
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = torch.unsqueeze(labels, 2)  # 在第 2 个维度上增加一个维度
            labels = torch.unsqueeze(labels, 3)  # 在第 3 个维度上增加一个维度
            labels = labels.expand(-1, -1, 180, 320)  # 扩展张量以匹配输出大小
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    # 保存模型
    torch.save(model.state_dict(), 'D:/data/model/u-netModel.pth')


# 实例化模型
model1 = UNet()
train_labels1 = [0] * 142
folder_path1 = r'D:\data\test1'
for i in range(42, 61):
    train_labels1[i] = 1
train_model(folder_path1, train_labels1, model1)

train_labels2 = [0] * 230
folder_path2 = r'D:\data\test2'
for i in range(171, 198):
    train_labels2[i] = 1
model2 = UNet()
model2.load_state_dict(torch.load('D:/data/model/u-netModel.pth'))
train_model(folder_path2, train_labels2, model2)

train_labels3 = [0] * 278
folder_path3 = r'D:\data\test3'
for i in range(15, 42):
    train_labels3[i] = 1
for i in range(255, 271):
    train_labels3[i] = 1
model3 = UNet()
model3.load_state_dict(torch.load('D:/data/model/u-netModel.pth'))
train_model(folder_path3, train_labels3, model3)

train_labels4 = [0] * 380
folder_path4 = r'D:\data\test4'
for i in range(8, 15):
    train_labels4[i] = 1
for i in range(173, 180):
    train_labels4[i] = 1
for i in range(207, 226):
    train_labels4[i] = 1
model4 = UNet()
model4.load_state_dict(torch.load('D:/data/model/u-netModel.pth'))
train_model(folder_path4, train_labels4, model4)

folder_path5 = r'D:\data\test5'
train_labels5 = [0] * 230
train_labels5[1] = 1
for i in range(119, 142):
    train_labels5[i] = 1
for i in range(161, 175):
    train_labels5[i] = 1
for i in range(190, 203):
    train_labels5[i] = 1
model5 = UNet()
model5.load_state_dict(torch.load('D:/data/model/u-netModel.pth'))
train_model(folder_path5, train_labels5, model5)
