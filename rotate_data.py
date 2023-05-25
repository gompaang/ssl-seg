import random

import torch
import torchvision
import torchvision.transforms as transforms


def rotate_img(img, rot):
  image = transforms.ToPILImage()(img)
  if rot == 0: #0
    return img
  elif rot == 1: #90
    r_img = image.rotate(90)
    return transforms.ToTensor()(r_img)
  elif rot == 2: #180
    r_img = image.rotate(180)
    return transforms.ToTensor()(r_img)
  elif rot == 3: #270
    r_img = image.rotate(270)
    return transforms.ToTensor()(r_img)
  else:
    raise ValueError('rotation should be 0, 90, 180, 270 degrees')


class CIFAR10Rotation(torchvision.datasets.CIFAR10):
  def __init__(self, root, train, download, transform):
    super().__init__(root=root, train=train, download=download, transform=transform)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    image, cls_label = super().__getitem__(index)

    # randomly image rotation
    rotation_label = random.choice([0, 1, 2, 3])
    image_rotated = rotate_img(image, rotation_label)
    rotation_label = torch.tensor(rotation_label).long()
    cls_label = torch.tensor(cls_label).long()  # cls_label
    return image, image_rotated, rotation_label, cls_label

