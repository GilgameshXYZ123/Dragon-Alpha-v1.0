import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__();
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, 1, 1, 0),
            nn.LeakyReLU(inplace=True)
        );

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv2d(ch3x3red, ch3x3, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        );

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv2d(ch5x5red, ch5x5, 5, 1, 2),
            nn.LeakyReLU(inplace=True)
        );

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(in_channels, pool_proj, 1, 1, 0),
            nn.LeakyReLU(inplace=True)
        );

    def forward(self, X):
        X1 = self.branch1.forward(X);
        X2 = self.branch2.forward(X);
        X3 = self.branch3.forward(X);
        X4 = self.branch4.forward(X);
        return torch.cat([X1, X2, X3, X4], dim=1);


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__();
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2);
        self.conv2 = nn.Conv2d(64, 192, 3, 1, 1);
        self.bn1 = nn.BatchNorm2d(192);

        self.inception3a = Inception(192,  64,  96, 128, 16, 32, 32);
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64);
        self.bn2 = nn.BatchNorm2d(480);

        self.inception4a = Inception(480, 192,  96, 208, 16,  48,  64);
        self.inception4b = Inception(512, 160, 112, 224, 24,  64,  64);
        self.inception4c = Inception(512, 128, 128, 256, 24,  64,  64);
        self.inception4d = Inception(512, 112, 144, 288, 32,  64,  64);
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128);

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128);
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128);

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1, inplace=True),
            nn.Linear(1024, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, 10),
        );

    def forward(self, X):
        X = self.conv1(X);
        X = F.leaky_relu(X);
        X = F.max_pool2d(X, 2);

        X = self.conv2(X);
        X = self.bn1(X);
        X = F.leaky_relu(X);
        X = F.max_pool2d(X, 2);

        X = self.inception3a(X);
        X = self.inception3b(X);
        X = self.bn2(X);
        X = F.max_pool2d(X, 2);

        X = self.inception4a(X);
        X = self.inception4b(X);
        X = self.inception4c(X);
        X = self.inception4d(X);
        X = self.inception4e(X);
        X = F.max_pool2d(X, 2);

        X = self.inception5a(X);
        X = self.inception5b(X);
        X = F.adaptive_avg_pool2d(X, 1);

        X = self.classifier(X.view(X.size(0), -1));
        return X;


class config(object):
    batchSize = 512;
    lr = 0.001;
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight = "C:\\Users\\Gilgamesh\\Desktop\\torch_cifar10_googlenet.pt";


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
    net = GoogLeNet(); print(net);
    net.to(config.device);
    net.train();

    opt = torch.optim.Adam(net.parameters(), lr=config.lr);
    loss_func = nn.CrossEntropyLoss();

    batch_count = 0;
    start_time = time.time();
    for epoch in range(epochs):
        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.to(config.device);
            label = label.to(config.device);

            output = net(data);
            loss = loss_func(output, label);
            if(batch_count % 10 == 0):
                print('epoch: {}, idx: {}, loss: {}'.format(epoch, batch_count, loss.item()))

            loss.backward();
            opt.step()
            opt.zero_grad();
            batch_count += 1;
        print("\n");

    div = time.time() - start_time;
    print('time: {} sec'.format(div))
    eachS = 1.0 * div / (config.batchSize * batch_count) * 1000;
    print("for each sample:", eachS);

    torch.save(net.state_dict(), config.weight);


def test():
    net = GoogLeNet();
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

    correct /= length;
    print("accuracy: ", correct);


if __name__ == '__main__':
    training(25);
    # test();