import torch
import torchvision
import torchvision.transforms as transforms
from torch import Tensor

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__();
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False);
        self.bn1 = nn.BatchNorm2d(out_channel);
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False);
        self.bn2 = nn.BatchNorm2d(out_channel);

        self.downsample = None;
        if(stride != 1 or in_channel != out_channel):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_channel)
            );

    def forward(self, X):
        res = X;

        X = self.bn1(self.conv1(X));
        X = F.leaky_relu(X, inplace=True);
        X = self.bn2(self.conv2(X));

        if(self.downsample != None): res = self.downsample(res);

        X = X + res;
        X = F.leaky_relu(X, inplace=True);
        return X;


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.prepare = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        );

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1)
        );

        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, 2),
            BasicBlock(128, 128, 1),
            BasicBlock(128, 128, 1),
            BasicBlock(128, 128, 1)
        );

        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, 2),
            BasicBlock(256, 256, 1),
            BasicBlock(256, 256, 1),
            BasicBlock(256, 256, 1),
            BasicBlock(256, 256, 1),
            BasicBlock(256, 256, 1)
        );

        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, 2),
            BasicBlock(512, 512, 1),
            BasicBlock(512, 512, 1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1);

        self.fc = nn.Sequential(
            nn.Dropout(0.1, inplace=True),
            nn.Linear(512, 256, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 10, bias=True)
        );

    def forward(self, X:Tensor):
        X = self.prepare(X);

        X = self.layer1(X);
        X = self.layer2(X);
        X = self.layer3(X);
        X = self.layer4(X);
        X = self.pool(X);

        X = self.fc(X.view(X.size(0), -1));
        return X;


class config(object):
    batchSize = 512;
    lr = 0.001;
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight = "C:\\Users\\Gilgamesh\\Desktop\\torch_cifar10_resnet18.pt";


print('device: {}'.format(config.device), );
print(torch.cuda.get_device_name(config.device));

train_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=False,
                                         transform=transforms.ToTensor())
test_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=False,
                                        transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batchSize, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def training(epochs):
    net = ResNet34(); print(net);
    net.to(config.device);
    net.train();

    opt = torch.optim.Adam(net.parameters(), lr=config.lr);
    # opt = torch.optim.SGD(net.parameters(), lr=config.lr);
    loss_func = nn.CrossEntropyLoss();

    batch_count = 0;
    start_time = time.time();
    for epoch in range(epochs):
        print();
        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.to(config.device);
            label = label.to(config.device);

            output = net(data);
            loss = loss_func(output, label);
            if(batch_count % 10 == 0):
                print('epoch: {}, batch: {}, loss: {}'.format(epoch, batch_count, loss.item()))

            loss.backward();
            opt.step()
            opt.zero_grad();
            batch_count += 1;

    div = time.time() - start_time;
    print('time: {} sec'.format(div))
    eachS = 1.0 * div / (config.batchSize * batch_count) * 1000;
    print("for each sample:", eachS);

    torch.save(net.state_dict(), config.weight);


def test():
    net = ResNet34();
    net.to(config.device);
    net.load_state_dict(torch.load(config.weight))
    net.eval();

    loss_func = nn.CrossEntropyLoss();

    avg_loss = 0.0;
    correct = 0.0;
    for batch_idx, (data, label) in enumerate(train_loader):
    # for batch_idx, (data, label) in enumerate(test_loader):
        data = data.to(config.device);
        label = label.to(config.device);

        output = net(data);
        loss = loss_func(output, label);
        ls = loss.item();
        avg_loss += ls;
        print('idx: {}, loss: {}'.format(batch_idx, ls))

        pred = torch.max(output.data, 1)[1];
        correct += (pred == label).sum();

    length = float(len(train_loader.dataset));
    # length = float(len(test_loader.dataset));

    avg_loss /= length;
    correct /= length;

    print("average loss: ", avg_loss)
    print("accuracy: ", correct);

if __name__ == '__main__':
    # 25 epochs for Adam
    # 50 epochs form SGD
    training(50);
    # test();