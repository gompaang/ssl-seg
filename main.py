import time
import argparse
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.models import resnet18

from rotate_data import *


def train(model, criterion, optimizer, num_epochs, lr):
    start_time = time.time()
    best_loss = 100

    for epoch in range(num_epochs):
        running_loss = 0.0

        model.train()

        for i, (imgs, imgs_rotated, rotation_label, cls_label) in enumerate(trainloader, 0):
            imgs, labels = imgs.to(device), cls_label.to(device)
            optimizer.zero_grad()
            predict = model(imgs)
            loss = criterion(predict, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / len(trainloader):.3f}')
        #wandb.log({'Epoch': epoch, 'loss': running_loss / len(trainloader)})

        avg_test_loss = 0.0
        with torch.no_grad():
            for imgs, imgs_rotated, labels, cls_labels in testloader:
                imgs, labels = imgs.to(device), cls_labels.to(device)
                predict = model(imgs)
                loss = criterion(predict, labels)
                avg_test_loss += loss

        if best_loss > (avg_test_loss / len(testloader)):
            best_loss = (avg_test_loss / len(testloader))

        print(f'Average Loss: {avg_test_loss / len(testloader):.3f}')
        wandb.log({'Epoch': epoch, 'loss': running_loss / len(trainloader)})

    total_time = time.time() - start_time
    print('Finish Training')
    print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
    print('Best val loss: {:4f}'.format(best_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=int, default=0.01)

    config = parser.parse_args()

    # 0. args
    num_epochs = config.num_epoch
    batch_size = config.batch_size
    num_workers = config.num_workers
    lr = config.lr
    wandb.init(project='ssl-cam', entity='heystranger')  # wandb

    # 1. data load
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = CIFAR10Rotation(
        root='./data', train=True, download=True, transform=transform_train)
    testset = CIFAR10Rotation(
        root='./data', train=False, download=True, transform=transform_test)

    trainloader = DataLoader(
        dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(
        dataset=testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    rot_classes = ('0', '90', '180', '270')

    # 2. model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet18(num_classes=len(classes))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    # 3. train
    train(model, criterion, optimizer, num_epochs=num_epochs, lr=lr)

