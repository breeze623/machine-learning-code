import torch
from torchvision import datasets
from torchvision import transforms
from torch.nn import functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

import warnings

warnings.filterwarnings('ignore')

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 定义前三层卷积层以及其中的池化层
        self.convlayer = nn.Sequential(
            # input:[b, 1, 28, 28] --> c1:[b, 6, 14, 14]
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            # c1:[b, 6, 14, 14] --> c2:[b, 16, 5, 5]
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            # c2:[b, 16, 5, 5] --> c3:[b, 120, 1, 1]
            nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
        )
        # 中间需要一个view操作，将dim=4的tensor转化为dim=2的tensor(vector)
        self.linearlayer = nn.Sequential(
            # input:[b, 120]
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, input):
        # input:[b, 1, 28, 28]
        batch_size = input.size(0)
        # 通过卷积层
        temp = self.convlayer(input)
        # 进行dim变换
        temp = temp.view(batch_size, 120)
        # 通过全连接层
        logits = self.linearlayer(temp)

        return logits

def main():
    # 初始化网络
    device = torch.device('cuda:0')
    net = LeNet5().to(device)

    # 超参数定义
    batch_size = 128
    epochs = 10
    lr = 0.01

    # 导入数据集
    train_dataset = datasets.MNIST('MNIST', True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST('MNIST', False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 测试数据是否读取正常
    # for index, (input, target) in enumerate(train_loader):
    #     print('input:', input.shape, 'target:', target.shape)
    #     break

    # 创建优化器
    optimizer = optim.SGD(net.parameters(), lr=lr)
    # 创建交叉熵损失函数
    CE_Loss = nn.CrossEntropyLoss().to(device)

    # 进行训练
    for epoch in range(epochs):
        print(f'Training the {epoch+1}th epoch')
        for batch_index, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)
            # 通过网络得到logits
            logits = net(input)
            # 计算损失
            loss = CE_Loss(logits, target)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            if batch_index % 100 == 0:
                print('{} batches have trained!'.format(batch_index))

        # 测试
        print('Testing')
        test_loss = 0
        correct_num = 0

        for idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            # data:[b, 1, 28, 28]  label:[b]
            temp = net(data)
            test_loss += CE_Loss(temp, label)

            predict = F.softmax(temp)       # predict:[b, 10]
            predict_class = predict.argmax(dim=1) # [b, 1]
            correct_num += predict_class.eq(label).float().sum().item()

        acc = correct_num/len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        print('Test Loss:{:.4f}, Accuracy:{:.2f}%'.format(test_loss, 100*acc))


if __name__ == '__main__':
    main()
