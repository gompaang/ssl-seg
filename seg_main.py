import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50
from torchvision.datasets import VOCSegmentation

import numpy as np
from PIL import Image


# data set
class VOCSegDataset(VOCSegmentation):
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        label[label > 20] = 0  # 클래스 인덱스 20 이상을 0으로 변경
        return image, label


if __name__ == '__main__':

    image_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-5, 5)),
        transforms.ToTensor(),
    ])

    target_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-5, 5)),
        transforms.ToTensor(),
    ])

    train_ds = VOCSegDataset(
        root='./data', year='2012', image_set='train', download=True,
        transform=image_transforms, target_transform=target_transforms)
    val_ds = VOCSegDataset(
        root='./data', year='2012', image_set='val', download=True,
        transform=image_transforms, target_transform=target_transforms)

    np.random.seed(0)
    num_classes = 21
    COLORS = np.random.randint(0, 2, size=(num_classes + 1, 3), dtype='uint8')

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

    fcn_model = fcn_resnet50(pretrained=False, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(fcn_model.parameters(), lr=0.001)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fcn_model.to(device)

    for epoch in range(10):  # 예시로 10 에폭으로 설정
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            # images = torch.tensor(labels)
            # labels = torch.tensor(labels)

            optimizer.zero_grad()
            outputs = fcn_model(images)['out']
            labels = labels.squeeze(1).long()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/10], Loss: {loss.item()}')
        print('finish')



