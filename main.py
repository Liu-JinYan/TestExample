import  torch
from  torch import nn
from  d2l import  torch as d2l
device=torch.device('mps')
n_train=50
x_train,_ = torch.sort(torch.rand(n_train)*5)

why=torch.sort(torch.rand(n_train)*5)
def f(x):
    return  2* torch.sin(x)+x**0.8

y_train=f(x_train)+torch.normal(0.0,0.5,(n_train,))
x_test = torch.arange(0,5,0.1)
y_truth = f(x_test)
n_test=len(x_test)

def plot_kernel_reg(y_hat):
    d2l.plot(x_test,[y_truth, y_hat], 'x', 'y', legend=['Truth'],xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha = 0.5)
    d2l.plt.show()

y_hat =torch.repeat_interleave(y_train.mean(),n_test)
plot_kernel_reg(y_hat)
#
# #非参数注意力汇聚
# x_repeat =x_test.repeat_interleave (n_train).reshape((-1, n_train))
# attention_weights = nn.functional.softmax(-(x_repeat - x_train)**2)
# y_hat=torch.matmul(attention_weights, y_train)
# plot_kernel_reg(y_hat)

print("test")