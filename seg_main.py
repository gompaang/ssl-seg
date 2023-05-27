import random
import time
import wandb
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
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

    for epoch in range(num_epochs):  # 예시로 10 에폭으로 설정
        running_loss = 0.0
        model.train()

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = fcn_model(images)['out']
            labels = labels.squeeze(1).long()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/len(train_loader):.3f}')

        avg_test_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = fcn_model(images)['out']
                labels = labels.squeeze(1).long()

                loss = criterion(outputs, labels)
                avg_test_loss += loss.item()

        if best_loss > (avg_test_loss / len(val_loader)):
            best_loss = (avg_test_loss / len(val_loader))

        print(f'Average Loss: {avg_test_loss / len(val_loader):.3f}')
        wandb.log({'Epoch': epoch, 'loss': avg_test_loss / len(val_loader)})

    total_time = time.time() - start_time
    print('Finish Training')
    print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
    print('Best val loss: {:4f}'.format(best_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=int, default=0.01)
    parser.add_argument('--ptmodel', type=str, default='basic')
    parser.add_argument('--path', type=str, default='./fcn-s.pth.tar')
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

    if ptmodel == 'rotation':
        # 2. model
        resnet_model = resnet50()
        resnet_model.load_state_dict(torch.load(model_path))

        fcn_model = fcn_resnet50(pretrained=False, num_classes=num_classes)
        state_dict_fcn = fcn_model.backbone.state_dict()
        state_dict_resnet = resnet_model['state_dict']

        backbone_state_dict = {
            k: v for k, v in state_dict_resnet.items() if k in state_dict_fcn
        }
        state_dict_fcn.update(backbone_state_dict)
        fcn_model.backbone.load_state_dict(state_dict_fcn)

        # fcn_model.backbone.load_state_dict(resnet50)
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(fcn_model.parameters(), lr=lr)

        # 3. train
        train(fcn_model, criterion, optimizer, num_epochs, lr)
        torch.save(fcn_model.state_dict(), path)

    elif ptmodel == 'basic':
        # 2. model
        fcn_model = fcn_resnet50(pretrained=True, num_classes=num_classes)
        fcn_model.to(device)
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(fcn_model.parameters(), lr=lr)

        # 3. train
        train(fcn_model, criterion, optimizer, num_epochs, lr)
        torch.save(fcn_model.state_dict(), path)



