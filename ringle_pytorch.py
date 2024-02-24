import torch
import numpy as np
import matplotlib.pyplot as plt


data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(x_data)


shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


ones = torch.ones(2, 3)
print(ones)

twos = torch.ones(2, 3) * 2 # 모든 원소에 2를 곱합니다.
print(twos)

threes = ones + twos       # shape이 비슷하기 때문에 더할 수 있습니다.
print(threes)              # tensor의 원소별 더한 값이 결과로 나옵니다.
print(threes.shape)        # 입력 tensor와 동일한 차원을 가지고 있습니다.

r1 = torch.rand(2, 3)
r2 = torch.rand(3, 2)
# 런타임 오류를 발생시키려면 아래 줄의 주석을 해제합니다.
# r3 = r1 + r2    size가 달라서 런타임 에러가 뜸


###############################################
#LeNet-5
import torch.nn as nn            # torch.nn.Module의 경우 PyTorch model의 부모 객체
import torch.nn.functional as F  # 활성화 함수 모듈 가져오기
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 입력 이미지 채널, 6개의 output 채널, 5x5 정방 합성곱 커널을 사용합니다.
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 아핀 변환: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 이미지 차원
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 최대 풀링은 (2, 2) 윈도우 크기를 사용합니다.
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 크기가 정방 사이즈인 경우, 단일 숫자만 지정할 수 있습니다.
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 크기는 배치 차원을 제외한 모든 차원을 가져옵니다.
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
net = LeNet()
print(net)                         # 인스턴스한 객체를 출력하면 어떤 값을 보여줄까요?

input = torch.rand(1, 1, 32, 32)   # 32x32 크기의 1채널의 흑백 이미지를 만듭니다.
print('\n이미지 배치 shape:')
print(input.shape)

output = net(input)                # 객체로부터 직접 forward() 함수를 호출하지 않습니다.
print('\n:결과 값')
print(output)
print(output.shape)

import torch
import torchvision
import torchvision.transforms as transforms

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)
    import matplotlib.pyplot as plt
    import numpy as np

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def imshow(img):
        img = img / 2 + 0.5     # 역정규화
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


    # 무작위로 학습 이미지 추출하기
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # 이미지 보여주기
    imshow(torchvision.utils.make_grid(images))
    # labels 출력하기
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

if __name__ == '__main__':
    main()

