import torch
from torch.nn import functional as F
from torchvision import transforms, datasets
from loguru import logger
from tqdm import tqdm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch.optim import lr_scheduler
from torch.nn.parameter import Parameter
from torchdiffeq import odeint

import os
import math
import torch.optim as optim
import torchvision

class Config:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = "cifar10"#mnist,cifar10
    epoch = 300
    batchsize = 256
    num_workers = 8
    hidden_dim = 4096
    torch.manual_seed(42)
    log_name = "0728_256_4096_1e-4_adam.log"
    best_acc = 0
    best_epoch = 0
    alpha = 1e-4
    mode = "nmODE"#nmODE,origin
    opt = "Adam"#Adam,SGD
    num_classes = 10
    input_dim = 256
    momentum = 0.9
    weight_decay = 5e-4
    delta = 1.0

config = Config()

class Bottleneck(nn.Module):

    expansion = 4
    
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        zip_channels = self.expansion * growth_rate
        self.features = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, zip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(zip_channels),
            nn.ReLU(True),
            nn.Conv2d(zip_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )
        
    def forward(self, x):
        out = self.features(x)
        out = torch.cat([out, x], 1)
        return out
    
class Transition(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.features = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2)
        )
        
    def forward(self, x):
        out = self.features(x)
        return out

class DenseNet(nn.Module):

    def __init__(self, num_blocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.reduction = reduction
        
        num_channels = 2 * growth_rate
        
        self.features = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        self.layer1, num_channels = self._make_dense_layer(num_channels, num_blocks[0])
        self.layer2, num_channels = self._make_dense_layer(num_channels, num_blocks[1])
        self.layer3, num_channels = self._make_dense_layer(num_channels, num_blocks[2], transition=False)
        self.avg_pool = nn.Sequential(
            nn.BatchNorm2d(num_channels),
            nn.ReLU(True),
            nn.AvgPool2d(8),
        )
        self.classifier = nn.Linear(num_channels, config.input_dim)
        
        self._initialize_weight()
        
    def _make_dense_layer(self, in_channels, nblock, transition=True):
        layers = []
        for i in range(nblock):
            layers += [Bottleneck(in_channels, self.growth_rate)]
            in_channels += self.growth_rate
        out_channels = in_channels
        if transition:
            out_channels = int(math.floor(in_channels * self.reduction))
            layers += [Transition(in_channels, out_channels)]
        return nn.Sequential(*layers), out_channels
    
    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        out = self.features(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

class neuralODE(torch.nn.Module):
    def __init__(self):
        super(neuralODE, self).__init__()
    def fresh(self, gamma):
        self.gamma = gamma
    def forward(self, t, y):
        dydt = -y + torch.pow(torch.sin(y+self.gamma),2)
        return dydt

class nmODEblock(torch.nn.Module):
    def __init__(self):
        super(nmODEblock, self).__init__()
        self.densenet40 = DenseNet([12,12,12], growth_rate=12)
        self.LP1 = torch.nn.Linear(config.input_dim,config.hidden_dim)
        self.LP2 = torch.nn.Linear(config.input_dim,config.hidden_dim)
        self.LP3 = torch.nn.Linear(config.hidden_dim,10)
        self.neuralODE = neuralODE()
        self.t = torch.tensor([0, config.delta], device=config.device)
    def forward(self,x,y):
        x = self.densenet40(x)
        gamma = self.LP1(x)
        self.neuralODE.fresh(gamma)
        y = odeint(self.neuralODE, y, self.t)[-1]
        z = self.LP3(self.LP2(x)*y)
        return z,y

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if config.dataset=="mnist":
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./dataset', train=True, transform=train_transform, download=True),
        batch_size=config.batchsize,
        shuffle=True,
        num_workers=config.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./dataset', train=False, transform=test_transform, download=True),
        batch_size=config.batchsize,
        shuffle=False,
        num_workers=config.num_workers,
    )
elif config.dataset=="cifar10":
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./dataset', train=True, transform=train_transform, download=False),
        batch_size=config.batchsize,
        shuffle=True,
        num_workers=config.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./dataset', train=False, transform=test_transform, download=False),
        batch_size=config.batchsize,
        shuffle=False,
        num_workers=config.num_workers,
    )

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == "__main__":

    logger.add(config.log_name)

    # net = DenseNet([32,32,32], growth_rate=12).to(config.device)
    net = nmODEblock().to(config.device)

    criterion=torch.nn.CrossEntropyLoss(size_average=True).to(config.device)
    if config.opt=="Adam":
        optimizer=torch.optim.Adam(net.parameters(),lr=config.alpha)
        scheduler=lr_scheduler.CosineAnnealingLR(optimizer,config.epoch)
    elif config.opt=="SGD":
        optimizer = optim.SGD(net.parameters(), lr=config.alpha, momentum=config.momentum, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225])

    for epoch_id in range(config.epoch):
        # train
        net.train()
        with tqdm(len(train_loader)) as pbar:
            for batch_id, sample in enumerate(train_loader):
                optimizer.zero_grad()
                data, label= sample[0].to(config.device), sample[1].to(config.device)
                y0 = torch.zeros(data.shape[0], config.hidden_dim, device=config.device)
                z,_ = net(data,y0)
                # z = net(data)
                cost=criterion(z,label)
                cost.backward()
                optimizer.step()
                batch_acc=(z.argmax(axis=1) == label).sum()/label.shape[0]
                pbar.update(1)
                pbar.set_description("Epoch: {}, Batch: {}/{}, Train Acc: {:.5f}".format(epoch_id, batch_id, len(train_loader), batch_acc))
        #test
        with torch.no_grad():
            net.eval()
            total, correct = 0.0, 0.0
            for batch_id, sample in enumerate(test_loader):
                data, label = sample[0].to(config.device), sample[1]
                y1 = torch.zeros(data.shape[0], config.hidden_dim, device=config.device)          
                a_pred,_ = net(data,y1)
                # a_pred = net(data)
                a_pred = a_pred.cpu().argmax(dim=1)
                a_true = label
                correct += (a_pred == a_true).sum().item()
                total += a_true.shape[0]
            scheduler.step()
            logger.info(f"Epoch: {epoch_id} Test acc={correct/total:.4f}")
            if config.best_acc<correct/total:
                config.best_acc=correct/total
                config.best_epoch=epoch_id
            logger.info(f"Best Epoch: {config.best_epoch} Best acc={config.best_acc:.4f}")