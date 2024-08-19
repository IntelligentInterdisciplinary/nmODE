import torch
from torch.utils.data import DataLoader
from data.data_loader import Dataset_ETT_hour
from torchdiffeq import odeint


class Config:
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size=32
    seq_len = 336
    label_len = 0
    pred_len = 96
    epoch = 100
    lr=0.00001
    num_workers=8
    inputsize=1
    ysize=40960
    outputsize=1
    delta=0.05
config=Config()

dataset_train = Dataset_ETT_hour(flag='train',size=(config.seq_len,config.label_len,config.pred_len))
dataset_test = Dataset_ETT_hour(flag='test',size=(config.seq_len,config.label_len,config.pred_len))
dataloader_train = DataLoader(
            dataset_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True)
dataloader_test = DataLoader(
            dataset_test,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=True)

class neuralODE(torch.nn.Module):
    def __init__(self):
        super(neuralODE, self).__init__()
        self.gamma = None
    def fresh(self, gamma):
        self.gamma = gamma
    def forward(self, t, y):
        dydt = -y + torch.pow(torch.sin(y+self.gamma),2)
        return dydt

class Net(torch.nn.Module): #nmODE
    def __init__(self):
        super(Net, self).__init__()
        self.LP1=torch.nn.Linear(config.inputsize,config.ysize)
        self.LP2=torch.nn.Linear(config.inputsize,config.ysize)
        self.LP3=torch.nn.Linear(config.ysize,config.outputsize)
        self.neuralODE = neuralODE()
        self.t = torch.tensor([float(0), float(config.delta)])

        self.gamma=None
        self.y=None

        torch.nn.init.orthogonal_(self.LP1.weight)
        torch.nn.init.orthogonal_(self.LP2.weight)
        torch.nn.init.orthogonal_(self.LP3.weight)

    def forward(self,x,y):
        self.gamma=self.LP1(x)
        self.neuralODE.fresh(self.gamma)
        self.y=odeint(self.neuralODE, y, self.t,method="rk4",options=dict(step_size=config.delta))[-1]
        z=self.LP3(self.LP2(x)*self.y)
        return z,self.y

net=Net()
net=net.to(config.device)
optimizer=torch.optim.Adam(net.parameters(),lr=config.lr)
criterion=torch.nn.MSELoss(size_average=True).to(config.device)
best_loss=float('inf')


if __name__ == '__main__':
    for epoch in range(config.epoch):
        net.train()
        loss_mean=0
        for i,(src,trg) in enumerate(dataloader_train):
            optimizer.zero_grad()
            pred=[]
            src=src.to(config.device)
            trg=trg.to(config.device)
            src=src.permute(1,0,2).float()
            trg=trg.permute(1,0,2).float()
            y=torch.zeros(config.batch_size,config.ysize,requires_grad=True).to(config.device)
            for x in src:
                z,y=net(x,y)
            pred.append(z)
            for _ in range(trg.shape[0]-1):
                z,y=net(z,y)
                pred.append(z)
            pred=torch.cat(pred,dim=0)
            trg=trg.reshape(-1,1)
            loss=criterion(pred,trg)
            loss.backward()
            optimizer.step()
            loss_mean+=loss.item()
        loss_mean/=len(dataloader_train)
        print(f'Train:epoch:{epoch+1}/{config.epoch},loss:{loss_mean:.4f}')
        
        with torch.no_grad():
            net.eval()
            loss_mean=0
            for i,(src,trg) in enumerate(dataloader_test):
                pred=[]
                src=src.to(config.device)
                trg=trg.to(config.device)
                src=src.permute(1,0,2).float()
                trg=trg.permute(1,0,2).float()
                y=torch.zeros(config.batch_size,config.ysize,requires_grad=True).to(config.device)
                for x in src:
                    z,y=net(x,y)
                pred.append(z)
                for _ in range(trg.shape[0]-1):
                    z,y=net(z,y)
                    pred.append(z)
                pred=torch.cat(pred,dim=0)
                trg=trg.reshape(-1,1)
                loss=criterion(pred,trg)
                loss_mean+=loss.item()
            loss_mean/=len(dataloader_test)
            print(f'Test:epoch:{epoch+1}/{config.epoch},MSE_loss:{loss_mean:.4f}')
            if loss_mean<best_loss:
                best_loss=loss_mean
                print(f'[Best]:epoch:{epoch+1},MSE_loss:{loss_mean:.4f}')