import torch
import torchvision
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
import math
from torch.autograd import Variable
import os
import torch.optim as optim
# import matplotlib.pyplot as plt
# import numpy as np

class Config:
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    dataset="fashionmnist"#mnist,cifar10,fashionmnist
    epoch = 300
    batchsize = 200
    num_workers = 8
    torch.manual_seed(42)
    model_name = "noaug_abla0531_40960"
    log_name = f"{model_name}.log"
    best_acc = 0
    best_epoch = 0

    alpha = 1e-3
    weight_decay = 1e-3
    num_classes = 10
    input_dim = 28*28
    lam = 1.0
    rtol = 1e-3
    atol = 1e-3

    hidden_dim = 40960
    delta = 1.0

    opt = "Adam"#Adam,SGD
    T_max = 100
    momentum = 0.9
    
    scheduler = "multistep"#cosine,multistep
    milestones= [150, 225]

config = Config()

class neuralODE(torch.nn.Module):
    def __init__(self):
        super(neuralODE, self).__init__()
    def fresh(self, gamma):
        self.gamma = gamma
    def forward(self, t, y):
        dydt = -config.lam * y + torch.pow(torch.sin(y+self.gamma),2)
        return dydt

class nmODEblock(torch.nn.Module):
    def __init__(self):
        super(nmODEblock, self).__init__()
        self.LP1 = torch.nn.Linear(config.input_dim,config.hidden_dim)
        self.LP2 = torch.nn.Linear(config.input_dim,config.hidden_dim)
        self.LP3 = torch.nn.Linear(config.hidden_dim,10)
        self.neuralODE = neuralODE()
        self.t = torch.tensor([0,config.delta], device=config.device)
    def forward(self,x,y):
        x = x.view(-1,config.input_dim)
        gamma = self.LP1(x)
        self.neuralODE.fresh(gamma)
        y = odeint(self.neuralODE, y, self.t, method='dopri5', rtol=config.rtol, atol=config.atol)[-1]
        z = self.LP3(self.LP2(x) * y)
        return z,y

train_transform = transforms.Compose([
    # transforms.RandomCrop(28, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(15), 
    # transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
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
elif config.dataset=="fashionmnist":
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./dataset', train=True, transform=train_transform, download=True),
        batch_size=config.batchsize,
        shuffle=True,
        num_workers=config.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./dataset', train=False, transform=test_transform, download=True),
        batch_size=config.batchsize,
        shuffle=False,
        num_workers=config.num_workers,
    )

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == "__main__":

    model_save_directory = "models"
    if not os.path.exists(model_save_directory):
        os.makedirs(model_save_directory)
    model_save_path = os.path.join(model_save_directory, f"{config.model_name}_model_best.pth")

    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    log_file_path = os.path.join(log_directory, os.path.basename(config.log_name))
    logger.add(log_file_path)

    net = nmODEblock().to(config.device)
    criterion=torch.nn.CrossEntropyLoss(size_average=True).to(config.device)
    if config.opt == "Adam":
        optimizer=torch.optim.Adam(net.parameters(),lr=config.alpha,weight_decay=config.weight_decay)
    elif config.opt == "SGD":
        optimizer=torch.optim.SGD(net.parameters(), lr=config.alpha, weight_decay=config.weight_decay, momentum=config.momentum)
    if config.scheduler == "cosine":
        scheduler=lr_scheduler.CosineAnnealingLR(optimizer,config.T_max)
    elif config.scheduler == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.milestones)
    for epoch_id in range(config.epoch):
        # train
        net.train()
        with tqdm(len(train_loader)) as pbar:
            for batch_id, sample in enumerate(train_loader):
                optimizer.zero_grad()
                data, label= sample[0].to(config.device), sample[1].to(config.device)
                y0 = torch.zeros(data.shape[0], config.hidden_dim, device=config.device)
                z,_ = net(data,y0)
                cost=criterion(z,label)
                cost.backward()
                optimizer.step()
                batch_acc=(z.argmax(axis=1) == label).sum()/label.shape[0]
                pbar.update(1)
                pbar.set_description("Epoch: {}, Batch: {}/{}, Train Acc: {:.5f}, Train Loss: {:.5f}".format(epoch_id, batch_id, len(train_loader), batch_acc,cost.item()))
        #test
        with torch.no_grad():
            net.eval()
            test_loss, total, correct = 0.0, 0.0, 0.0
            for batch_id, sample in enumerate(test_loader):
                data, label = sample[0].to(config.device), sample[1].to(config.device)          
                y1 = torch.zeros(data.shape[0], config.hidden_dim, device=config.device)          
                a_pred,_ = net(data,y1)
                loss = criterion(a_pred, label) 
                test_loss += loss.item() * data.size(0) 
                a_pred = a_pred.cpu().argmax(dim=1)
                a_true = label.cpu()
                correct += (a_pred == a_true).sum().item()
                total += a_true.shape[0]
            test_loss = test_loss / total
            scheduler.step()
            logger.info(f"Epoch: {epoch_id} Test loss={test_loss:.4f} Test acc={correct/total:.4f}")
            if config.best_acc<correct/total:
                config.best_acc=correct/total
                config.best_epoch=epoch_id
                # 保存模型
                torch.save(net.state_dict(), model_save_path)
                logger.info(f"Model saved at {model_save_path} at epoch {epoch_id} with accuracy {config.best_acc:.4f}")
            logger.info(f"Best Epoch: {config.best_epoch} Best acc={config.best_acc:.4f}")
