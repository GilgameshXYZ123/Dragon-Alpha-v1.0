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


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__();
        self.prepare = nn.Sequential(
            nn.Conv2d(3, 96, 6, 2, 2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3, 2, 1)
        );

        self.layer1 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, 2, 1)
        );

        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.1, inplace=True),
            nn.Linear(256 * 2 * 2, 512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 10));

    def forward(self, X):
        X = self.prepare(X);
        X = self.layer1(X);
        X = self.layer2(X);
        X = self.classifier(X.view(X.size(0), -1));
        return X;


class config(object):
    batchSize = 512;
    lr = 0.001;
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight = "C:\\Users\\Gilgamesh\\Desktop\\torch_cifar10_alexnet.pt";


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
    net = AlexNet(); print(net);
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
    net = AlexNet();
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
    training(20);
    # test();