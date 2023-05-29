import time
import argparse
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models import resnet18
from torchvision.models import resnet50
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from rotate_data import *


def train(model, criterion, optimizer, num_epochs, lr, task):
    start_time = time.time()
    best_loss = 100
    avg_accuracy = 0

    for epoch in range(num_epochs):
        running_loss = 0.0

        model.train()

        for i, (imgs, imgs_rotated, rotation_label, cls_label) in enumerate(trainloader, 0):
            if task == 'rotation':
                imgs, labels = imgs_rotated.to(device), rotation_label.to(device)
            else:
                imgs, labels = imgs.to(device), cls_label.to(device)

            optimizer.zero_grad()
            predict = model(imgs)
            loss = criterion(predict, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/len(trainloader):.3f}')
        #wandb.log({'Epoch': epoch, 'loss': running_loss/len(trainloader)})

        avg_test_loss = 0.0

        model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for imgs, imgs_rotated, labels, cls_labels in testloader:
                if task == 'rotation':
                    imgs, labels = imgs_rotated.to(device), labels.to(device)
                else:
                    imgs, labels = imgs.to(device), cls_labels.to(device)

                predict = model(imgs)
                loss = criterion(predict, labels)
                avg_test_loss += loss

                _, predicted = torch.max(predict.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)

        if best_loss > (avg_test_loss / len(testloader)):
            best_loss = (avg_test_loss / len(testloader))

        if avg_accuracy < accuracy:
            avg_accuracy = accuracy

        print(f'Average Loss: {avg_test_loss/len(testloader):.3f}')
        print(f'Average accuracy: {accuracy:.3f}')
        wandb.log({'Accuracy': accuracy, 'loss': avg_test_loss/len(testloader)})

    total_time = time.time() - start_time
    print('Finish Training')
    print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    print('Average accuracy: {:4f}'.format(avg_accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=int, default=0.01)
    parser.add_argument('--task', type=str, default='rotation')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--path', type=str, default='./checkpoint.pth.tar')
    parser.add_argument('--ft_path', type=str, default='./resnet18-sslft.pth.tar')

    config = parser.parse_args()

    # 0. args
    num_epochs = config.num_epoch
    batch_size = config.batch_size
    num_workers = config.num_workers
    lr = config.lr
    task = config.task
    model_name = config.model
    path = config.path
    ft_path = config.ft_path
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if task == 'rotation':
        # 2. model
        if model_name == 'resnet18':
            model = resnet18(num_classes=len(rot_classes))
        elif model_name == 'resnet50':
            model = resnet50(num_classes=len(rot_classes))

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr)

        # 3. train
        train(model, criterion, optimizer, num_epochs=num_epochs, lr=lr, task=task)
        torch.save(model.state_dict(), path)

    elif task == 'classification':
        # 2. model
        if model_name == 'resnet18':
            model = resnet18(num_classes=4)
        elif model_name == 'resnet50':
            model = resnet50(num_classes=4)

        model.load_state_dict(torch.load(path))

        for param in model.parameters():
            param.requires_grad = False

        fc_in_features = model.fc.in_features
        model.fc = nn.Linear(fc_in_features, 10)

        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr)

        train(model, criterion, optimizer, num_epochs=num_epochs, lr=lr, task=task)
        torch.save(model.state_dict(), ft_path)

    elif task == 'supervised':
        # 2. model
        if model_name == 'resnet18':
            model = resnet18(num_classes=10)
        elif model_name == 'resnet50':
            model = resnet50(num_classes=10)

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr)

        # 3. train
        train(model, criterion, optimizer, num_epochs=num_epochs, lr=lr, task=task)
        torch.save(model.state_dict(), path)
