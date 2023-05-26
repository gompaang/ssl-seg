import torch
import torch.nn as nn
import torchvision.models as models


# 저장된 ResNet-50 모델 불러오기
checkpoint = torch.load('./checkpoint-r50.pth.tar', map_location=lambda storage, loc: storage)
resnet50_checkpoint = torch.load('./checkpoint-r50.pth.tar')  # 저장된 모델 경로에 맞게 수정
resnet = models.resnet50(pretrained=False)
resnet.load_state_dict(resnet50_checkpoint)

class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.resnet = resnet
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=1)
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x

# FCN 모델 인스턴스 생성
num_classes = 10  # 예시로 클래스 수를 10으로 설정
fcn_model = FCN(num_classes=num_classes)


# 입력 이미지
input_image = torch.randn(1, 3, 224, 224)  # 예시로 224x224 크기의 RGB 이미지를 사용

# FCN 네트워크에 이미지 전달
output = fcn_model(input_image)

# 결과 출력
print(output.shape)
