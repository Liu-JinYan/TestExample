import  torch
from  torch import  nn
from torch.utils.data import  DataLoader
from torchvision import  transforms
from  torchvision import  datasets
import  numpy as np

np.random.seed(123)
torch.manual_seed(123)
LR=0.0001

hidden_size=128
epochs =20
device=torch.device('mps')
train_dataset = datasets.FashionMNIST('./data',True,transforms.ToTensor(),download=False)
test_dataset = datasets.FashionMNIST('./data',False,transforms.ToTensor())

train_loader=DataLoader(train_dataset,256,True)
test_loader = DataLoader(test_dataset,256,False)

#mlp layer
num_input,num_outputs,num_hiddens = 256,10,hidden_size
mlp = nn.Sequential(
    nn.Linear(num_input,num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens,num_outputs)

).to(device)

#AutoEncoder

class AutoEncode(nn.Module):
    def __init__(self):
        super(AutoEncode,self).__init__()

        self.en_conv=nn.Sequential(
            nn.Conv2d(1,16,4,2,1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16,32,4,2,1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32,16,3,1,1),
            nn.BatchNorm2d(16),
            nn.Tanh()


        )

        self.en_fc=nn.Linear(16*7*7,hidden_size)
        self.de_fc=nn.Linear(hidden_size,16*7*7)
        self.de_conv=nn.Sequential(
            nn.ConvTranspose2d(16,16,4,2,1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.ConvTranspose2d(16,1,4,2,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        en = self.en_conv(x)
        code = self.en_fc(en.view(en.size(0),-1))
        de=self.de_fc(code)
        decode=self.de_conv(de.view(de.size(0)),16,7,7)
        return  code,decode

net = AutoEncode().to(device)

def AutoEncoder_train():



print("*************end************")








