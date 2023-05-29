import random
import time
import wandb
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.models.segmentation import fcn_resnet50
from torchvision.datasets import VOCSegmentation

import numpy as np
from PIL import Image


# data set
class VOCSegDataset(VOCSegmentation):
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        label[label > 20] = 0
        return image, label


def train(model, criterion, optimizer, num_epochs, lr):
    start_time = time.time()
    best_loss = 100
    best_iou = 0

    for epoch in range(num_epochs):  # 예시로 10 에폭으로 설정
        running_loss = 0.0
        model.train()

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)['out']  # fcn_model
            labels = labels.squeeze(1).long()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/len(train_loader):.3f}')

        model.eval()
        avg_test_loss = 0.0
        iou_list = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)['out']  # fcn_model
                labels = labels.squeeze(1).long()

                loss = criterion(outputs, labels)
                avg_test_loss += loss.item()

                iou = calculate_iou(outputs, labels, num_classes)
                iou_list.append(iou)


        miou = np.mean(iou_list)

        if best_loss > (avg_test_loss / len(val_loader)):
            best_loss = (avg_test_loss / len(val_loader))

        if best_iou < miou:
            best_iou = miou

        print(f'Average Loss: {avg_test_loss / len(val_loader):.3f}')
        print(f'Average iou: {miou:.3f}')
        wandb.log({'MIoU': miou, 'loss': avg_test_loss / len(val_loader)})

    total_time = time.time() - start_time
    print('Finish Training')
    print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    print('Best miou: {:4f}'.format(best_iou))


def calculate_iou(outputs, targets, num_classes):
    pred = outputs.argmax(dim=1)
    iou_list = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = targets == cls
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        iou = intersection / (union + 1e-10)  # avoid division by zero
        iou_list.append(iou)
    mean_iou = np.mean(iou_list)
    return mean_iou


def pixel_accuracy(pred, target):
    # 모델의 예측 클래스 가져오기
    _, pred_labels = torch.max(pred, dim=1)

    # 타깃과 예측 결과의 픽셀 단위 정확도 계산
    correct_pixels = torch.sum(pred_labels == target).item()
    total_pixels = target.numel()
    accuracy = correct_pixels / total_pixels

    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=int, default=0.01)
    parser.add_argument('--ptmodel', type=str, default='basic')
    parser.add_argument('--path', type=str, default='./fcn.pth.tar')
    parser.add_argument('--model_path', type=str, default='./r50-rc.pth.tar')

    config = parser.parse_args()

    # 0. args
    num_epochs = config.num_epoch
    batch_size = config.batch_size
    num_workers = config.num_workers
    lr = config.lr
    ptmodel = config.ptmodel
    path = config.path
    model_path = config.model_path
    wandb.init(project='ssl-cam', entity='heystranger')  # wandb

    # 1. data load
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

    train_set = VOCSegDataset(
        root='./data', year='2012', image_set='train', download=True,
        transform=image_transforms, target_transform=target_transforms)
    val_set = VOCSegDataset(
        root='./data', year='2012', image_set='val', download=True,
        transform=image_transforms, target_transform=target_transforms)

    # np.random.seed(0)
    # COLORS = np.random.randint(0, 2, size=(num_classes + 1, 3), dtype='uint8')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    num_classes = 21
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if ptmodel == 'modified':
        # 2. model
        resnet_model = resnet50(num_classes=10)
        resnet_model.load_state_dict(torch.load(model_path))
        fc_in_features = resnet_model.fc.in_features
        resnet_model.fc = nn.Linear(fc_in_features, 1000)
        fcn_model = fcn_resnet50(pretrained=False, num_classes=num_classes)

        state_dict = resnet_model.state_dict()
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        fcn_model.backbone.load_state_dict(state_dict)

        fcn_model = fcn_model.to(device)
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(fcn_model.parameters(), lr=lr)

        # 3. train
        train(fcn_model, criterion, optimizer, num_epochs, lr)
        torch.save(fcn_model.state_dict(), path)

    elif ptmodel == 'basic':
        # 2. model
        fcn_model = fcn_resnet50(pretrained=False, num_classes=num_classes)
        fcn_model.to(device)
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(fcn_model.parameters(), lr=lr)

        # 3. train
        train(fcn_model, criterion, optimizer, num_epochs, lr)
        torch.save(fcn_model.state_dict(), path)



