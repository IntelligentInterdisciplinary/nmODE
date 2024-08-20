import torch
import time
from torchdiffeq import odeint

class Config:
    device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    inputsize=1
    ysize=10000
    outputsize=1
    delta=1
config=Config()

class neuralODE(torch.nn.Module):
    def __init__(self):
        super(neuralODE, self).__init__()
        self.gamma = None
    def fresh(self, gamma):
        self.gamma = gamma
    def forward(self, t, y):
        dydt = -y + torch.pow(torch.sin(y+self.gamma),2)
        return dydt

class Net(torch.nn.Module):
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


if __name__ == '__main__':
    with torch.no_grad():
        net.eval()
        x=torch.zeros(1,config.inputsize,requires_grad=True).to(config.device)
        y=torch.zeros(1,config.ysize,requires_grad=True).to(config.device)
        

        start_time = time.time()
        z,y=net(x,y)
        end_time = time.time()
        print("=======================train_time:",end_time - start_time)