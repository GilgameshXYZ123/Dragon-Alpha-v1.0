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


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__();
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False);
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False);
        self.bn1 = nn.BatchNorm2d(64);

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1, bias=False);
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False);
        self.bn2 = nn.BatchNorm2d(128);

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1, bias=False);
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1, bias=False);
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1, bias=False);
        self.bn3 = nn.BatchNorm2d(256);

        self.conv8  = nn.Conv2d(256, 512, 3, 1, 1, bias=False);
        self.conv9  = nn.Conv2d(512, 512, 3, 1, 1, bias=False);
        self.conv10 = nn.Conv2d(512, 512, 3, 1, 1, bias=False);
        self.bn4 = nn.BatchNorm2d(512);

        self.conv11 = nn.Conv2d(512, 512, 3, 1, 1, bias=False);
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 1, bias=False);
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 1, bias=False);

        self.classifier = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 10));

    def forward(self, X):
        X = F.leaky_relu(self.conv1(X), inplace=True);
        X = F.leaky_relu(self.conv2(X), inplace=True);
        X = self.bn1(X);
        X = F.max_pool2d(X, 2);

        X = F.leaky_relu(self.conv3(X), inplace=True);
        X = F.leaky_relu(self.conv4(X), inplace=True);
        X = self.bn2(X);
        X = F.max_pool2d(X, 2);

        X = F.leaky_relu(self.conv5(X), inplace=True);
        X = F.leaky_relu(self.conv6(X), inplace=True);
        X = F.leaky_relu(self.conv7(X), inplace=True);
        X = self.bn3(X);
        X = F.max_pool2d(X, 2);

        X = F.leaky_relu(self.conv8(X), inplace=True);
        X = F.leaky_relu(self.conv9(X), inplace=True);
        X = F.leaky_relu(self.conv10(X), inplace=True);
        X = self.bn4(X);
        X = F.max_pool2d(X, 2);

        X = F.leaky_relu(self.conv11(X), inplace=True);
        X = F.leaky_relu(self.conv12(X), inplace=True);
        X = F.leaky_relu(self.conv13(X), inplace=True);
        X = F.max_pool2d(X, 2);

        X = self.classifier(X.view(X.size(0), -1));
        return X;


class config(object):
    batchSize = 512;
    lr = 0.001;
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight = "C:\\Users\\Gilgamesh\\Desktop\\torch_cifar10_vgg16.pt";


print('device: {}'.format(config.device));
print(torch.cuda.get_device_name(config.device));

train_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=False,
                                         transform=transforms.ToTensor())
test_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=False,
                                        transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batchSize, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def training(epochs):
    net = VGG16(); print(net);
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
    net = VGG16();
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