import enum
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

num_epochs = 8
num_classes = 2
batch_size = 128
learning_rate = 0.002

labels = pd.read_csv('train_labels.csv')
train_path = 'train/'
test_path = 'test/'

train, val = train_test_split(labels, stratify=labels['label'], test_size=0.1)


# dataset의 이미지가 너무 많아서 batch 단위로 가져와서 처리!!
class MyDataset(Dataset):
    def __init__(self, df_data, data_dir='./', transform=None):
        super().__init__()
        self.df = df_data.values
        self.data_dir = data_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name, label = self.df[index]
        img_path = os.path.join(self.data_dir, (img_name + '.tif'))
        image = cv2.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)

        return image, label

trans_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(64, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

trans_valid = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(64, padding_mode='reflect'),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset_train = MyDataset(df_data=train, data_dir=train_path, transform=trans_train)
dataset_valid = MyDataset(df_data=val, data_dir=train_path, transform=trans_valid)

loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
loader_valid = DataLoader(dataset=dataset_valid, batch_size=batch_size // 2, shuffle=False, num_workers=0)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=2) 

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg = nn.AvgPool2d(8)
        self.fc = nn.Linear(512 * 1 * 1 , 2)

    def forward(self, input):
        output = self.pool(F.leaky_relu(self.bn1(self.conv1(input))))
        output = self.pool(F.leaky_relu(self.bn2(self.conv2(output))))
        output = self.pool(F.leaky_relu(self.bn3(self.conv3(output))))
        output = self.pool(F.leaky_relu(self.bn4(self.conv4(output))))
        output = self.pool(F.leaky_relu(self.bn5(self.conv5(output))))
        output = self.avg(output)
        output = output.view(-1, 512 * 1 * 1)
        output = self.fc(output)
        return output

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(), lr=learning_rate)

total_step = len(loader_train)
for epoch in range(num_epochs):
    for i, data in enumerate(loader_train):
        images, labels = data
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for data in loader_valid:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 22003 test images: {} %'.format(100 * correct / total))

torch.save(model.state_dict(), 'Model/model_1st.ckpt')