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
from sklearn.metrics import confusion_matrix
from torchvision.transforms.functional import to_tensor, to_pil_image

import numpy as np
from PIL import Image


# data set
class VOCSegDataset(VOCSegmentation):
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            target *= 255
            target[target > 20] = 0

        return img, target


def train(model, criterion, optimizer, num_epochs, lr, path):
    start_time = time.time()
    best_loss = 100
    best_iou = 0
    best_accuracy = 0.0

    for epoch in range(num_epochs):  # 예시로 10 에폭으로 설정
        running_loss = 0.0
        model.train()

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)['out']  # fcn_model

            # loss = criterion(outputs, labels)
            loss = criterion(outputs, labels.squeeze(1).long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss/len(train_loader)

        if best_loss > avg_loss:
            best_loss = avg_loss
            torch.save(fcn_model.state_dict(), path)

        print(f'[{epoch + 1}] loss: {avg_loss:.3f}')
        print(f'Average Loss: {avg_loss:.3f}')

        model.eval()
        # running_iou = 0.0
        total_correct_pixels = 0
        total_pixels = 0
        total_iou = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)['out']  # fcn_model
                iou = compute_iou(outputs, labels)
                total_iou += iou

        avg_iou = total_iou / len(val_loader)
        # avg_accuracy = total_correct_pixels / total_pixels

        if best_iou < avg_iou:
            best_iou = avg_iou
        # if best_accuracy < avg_accuracy:
        #     best_accuracy = avg_accuracy

        print(f'avg iou: {iou:.3f}')
        # print(f'avg pixel accuracy: {avg_accuracy:.3f}')
        wandb.log({'iou': avg_iou, 'loss': avg_loss})
        # wandb.log({'pixel accuracy': avg_accuracy, 'loss': avg_loss})

    total_time = time.time() - start_time
    print('Finish Training')
    print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
    print('Best loss: {:4f}'.format(best_loss))
    # print('Best miou: {:4f}'.format(best_miou))
    print('Best accuracy: {:4f}'.format(best_accuracy))


def compute_iou(outputs, targets):
    predicted_labels = torch.argmax(outputs, dim=1)
    intersection = torch.logical_and(predicted_labels, targets).sum(dim=(1, 2))
    union = torch.logical_or(predicted_labels, targets).sum(dim=(1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()


def pixel_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct_pixels = (predicted == targets).sum().item()
    total_pixels = targets.numel()
    accuracy = correct_pixels / total_pixels
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
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
        root='./data', year='2012', image_set='train', download=False,
        transform=image_transforms, target_transform=target_transforms)
    val_set = VOCSegDataset(
        root='./data', year='2012', image_set='val', download=False,
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
        optimizer = optim.Adam(fcn_model.parameters(), lr=lr)

        # 3. train
        train(fcn_model, criterion, optimizer, num_epochs, lr, path)
        # torch.save(fcn_model.state_dict(), path)

    elif ptmodel == 'basic':
        # 2. model
        fcn_model = fcn_resnet50(pretrained=True, num_classes=num_classes)
        fcn_model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(fcn_model.parameters(), lr=lr)

        # 3. train
        train(fcn_model, criterion, optimizer, num_epochs, lr, path)
        # torch.save(fcn_model.state_dict(), path)



