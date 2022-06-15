import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision

from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

torch.set_default_tensor_type(torch.DoubleTensor)

def iris_type(s):
    it = {b'Iris-setosa':0,b'Iris-versicolor':1,b'Iris-virginica':2}
    return it[s]


class my_dataset(torch.utils.data.Dataset):
    # 对dataset函数重写，实现读取iris.data的数据
    def __init__(self, file_path='iris.data', train_or_test='train'):
        # 读入数据
        data = np.loadtxt(file_path, dtype=float, delimiter=",", converters={4: iris_type})
        self.x, self.y = np.split(data, (4,), axis=1)
        # 归一化
        scaler = MinMaxScaler()
        self.x = scaler.fit_transform(self.x)
        # one-hot编码
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(self.y)
        self.y = encoder.transform(self.y)

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, random_state=1, train_size=0.8)

        if train_or_test=='train':
            self.x = torch.from_numpy(x_train)
            self.y = torch.from_numpy(y_train)
        else:
            self.x = torch.from_numpy(x_test)
            self.y = torch.from_numpy(y_test)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class bpnn(nn.Module):
    def __init__(self):
        super(bpnn, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 6),
            nn.ReLU(inplace=True),
            nn.Linear(6, 3),
            nn.Softmax()
        )

    def forward(self, input):
        output = self.model(input)
        return output


# 使用GPU加速
device = torch.device('cuda:0')
net = bpnn().to(device)

# 导入数据集
train_dataset = my_dataset(train_or_test='train')
train_dataloader = DataLoader(train_dataset, batch_size=30, shuffle=True)
print('Train dataset load over!')

test_dataset = my_dataset(train_or_test='test')
test_dataloader = DataLoader(test_dataset, batch_size=30)
print('Tset dataset load over!')

# 测试数据集是否读取正常
# for index, (batch_x, batch_y) in enumerate(test_dataloader):
#     print(f"batch_id:{index}, batch_x.shape:{batch_x.shape}, batch_y.shape:{batch_y.shape}")
#     print(batch_x)
#     print(batch_y)
#     break

# 创建优化器
optimizer = optim.Adam(net.parameters(), lr=0.001)
# 创建交叉熵损失函数
criteon = nn.CrossEntropyLoss().to(device)



# 进行训练
print('Training...')
for epoch in range(500):
    for batch_index, (batch_x, batch_y) in enumerate(train_dataloader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        # 通过网络，得到预测值
        predict = net(batch_x)
        # 计算损失值
        loss = criteon(predict, batch_y)
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()

    if epoch%50==0:
        print('Testing')
        test_loss = 0
        acc = 0
        for index, (data, target) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)
            pre = net(data)
            test_loss += criteon(pre, target).item()

            target_index = target.argmax(dim=1, keepdim=True)
            pre_class = pre.argmax(dim=1, keepdim=True)
            acc += pre_class.eq(target_index).float().sum().item()

        test_loss /= len(test_dataloader.dataset)
        print('Test loss:{:.4f}, Accuracy:{}/{} ({:.0f}%)'.format(
            test_loss, acc, len(test_dataloader.dataset), 100. * acc / len(test_dataloader.dataset)
            )
        )



