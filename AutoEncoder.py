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

epochs1 =20


device=torch.device('mps')
train_dataset = datasets.FashionMNIST('./data',True,transforms.ToTensor(),download=False)
test_dataset = datasets.FashionMNIST('./data',False,transforms.ToTensor())

train_loader=DataLoader(train_dataset,256,True)
test_loader = DataLoader(test_dataset,256,False)

#mlp layer
num_input,num_outputs,num_hiddens = hidden_size,10,256
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
        decode=self.de_conv(de.view(de.size(0),16,7,7))
        return  code,decode

net = AutoEncode().to(device)

def AutoEncoder_train():
    net.train()

    optimizer= torch.optim.Adam(net.parameters(),lr=LR,weight_decay=5e-6)
    loss_f=nn.MSELoss()
    lr=LR;
    for epochs in range(1,epochs1+1):
        for step,(data,label) in enumerate(train_loader):
            data=data.to(device)
            net.zero_grad()
            code,decode=net(data)
            loss=loss_f(decode,data)
            loss.backward()
            optimizer.step()

        print('AutoEncoder epoch [%d/%d]  loss:%.4f'%(epochs1, epochs, loss))

    net.eval()

def train():
    AutoEncoder_train()

    print('mlp training**********************')
    optimizer =torch.optim.Adam(mlp.parameters(),lr=LR)
    loss=torch.nn.CrossEntropyLoss()

    #train mlp
    for epoch  in range(1,epochs1+1):
        mlp.train()

        for step,(data,lable) in enumerate(train_loader):
            data=data.to(device)
            lable =lable.to(device)

            mlp.zero_grad()

            code,decode=net(data)
            output=mlp(code)
            l =loss(output,lable)
            l.backward()
            optimizer.step()

        print('mlp :  eoch: [%d/%d], '%(epoch, epochs1), end='')

        test(test_loader)

def test(data_loader):

    mlp.eval()

    loss= torch.nn.CrossEntropyLoss()
    acc_sum,n,loss_sum=0,0,0.0
    for step,(data,lable) in enumerate(data_loader):
        data=data.to(device)
        lable=lable.to(device)
        code,decoder=net(data)
        output=mlp(code)
        l=loss(output,lable)
        acc_sum+=(output.argmax(dim=1)==lable).float().sum().item()
        n+=lable.shape[0]
        loss_sum+=l
    print('acc:%.2f%%  loss:%.4f' % (100 * acc_sum / n, loss_sum / (step + 1)))



if __name__ == '__main__':
    train()
    print('模型的测试精度: ', end='')
    test(test_loader)


print("*************end************")








