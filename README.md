# Semantic Segmentation using self-supervised learning

this is about Semantic Segmentation using supervised learning.

- self-supervised learning method
  - unsupervised representation learning by predicting image rotations
  - dataset: CIFAR10
  - model: ResNet-50
  - fine-tuning classification
- Semantic Segmentation
  - dataset: VOC 2012
  - model: fine-tuning self-supervised classifier(ResNet-50) + FCN


## abstract
현재 딥러닝 모델 성능은 데이터셋에 매우 의존적이다. 학습을 위한 데이터셋 구축과 라벨링 작업은 많은 시간과 비용이 들어간다. 이에 라벨이 없는 데이터를 통해서 학습할 수 있도록 하는 Self-supervised learning 연구의 중요성이 커지고 있다.

본 연구에서는 Self-supervised learning 을 통한 self-supervised classifier 로 Semantic Segmentation 을 수행하고, 성능을 확인하고자 한다. 

1) Image rotations 를 예측하는 pre-text task 에 대해 학습시킨 self-supervised classifier 에 대한 성능, 이를 fine-tuning 하여 성능, 기존 supervised learning 에 대한 성능을 비교한다.
2) 앞서 fine-tuning한 self-supervised classifier를 backbone 으로 하는 FCN 에 대한 성능, 기존 supervised learning 에 대한 성능을 비교한다. 


## experiment
1. Image Rotations (pre-text task) 실험
2. Semantic Segmentation 실험


## More details
[AJOU SOFTCON 2023-1](https://softcon.ajou.ac.kr/works/works.asp?uid=879) <br/><br/>
<img width="588" alt="포스터" src="https://github.com/gompaang/ssl-seg/assets/87194339/03dbac16-e358-4ae0-9875-6d0ad23fcacb">
