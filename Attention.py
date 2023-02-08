import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import copy
import  unittest

class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        super(Embeddings,self).__init__()
        self.lut=nn.Embedding(vocab,d_model)
        self.d_model=d_model
    def forward(self,x):
        return self.lut(x)*math.sqrt(self.d_model)

# embedding test block
d_model=512
vocab=1000
x=Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
emb = Embeddings(d_model, vocab)
embr=emb(x)
print(embr.shape)
print("***********************embedding test block")

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dropout=nn.Dropout(p=dropout)
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2)*-(math.log(1000)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe",pe)
    def forward(self,x):
        x=x+Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return  self.dropout(x)


# PositionalEncoding test block

dropout=0.1
max_len=60
pe=PositionalEncoding(d_model,dropout,max_len)
pe_result=pe(embr)
print(pe_result.shape)
plt.figure(figsize=(15,5))
pe=PositionalEncoding(20,0)
y=pe(Variable(torch.zeros(1,100,20)))
plt.plot(np.arange(100),y[0,:,4:8].data.numpy())
plt.legend(["dim %d"%p for p in [4,5,6,7]])
plt.show()
print("***********************PositionalEncoding test block")


def subsequent_mask(size):
    attn_shape=(1,size,size)
    subsequent_mask=np.triu(np.ones(attn_shape),k=1).astype('uint8')
    return  torch.from_numpy(1-subsequent_mask)

# subsequent_mask test block

sm=subsequent_mask(5)
print(sm)
plt.figure(figsize=(5,5))
plt.imshow(subsequent_mask(20)[0])
plt.show()
print("*********************** subsequent_mask test block")


#attention layer
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


#attention test block

query=key=value=pe_result
mask=Variable(torch.zeros(2,4,4))
attn,p_attn=attention(query,key,value,mask=mask)
print(attn,attn.shape,p_attn,p_attn.shape)
print("***********************#attention test block")

#multiheadedAttention

def clone(module,N):
    return  nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self,head,embedding_dim,dropout=0.1):
        super(MultiHeadedAttention,self).__init__()
        assert embedding_dim % head==0
        self.head=head
        self.d_k=embedding_dim//head
        self.linears=clone(nn.Linear(embedding_dim,embedding_dim),4)
        self.attn=None
        self.dropout=nn.Dropout(p=dropout)
    def forward(self,query,key,value,mask=None):
        if mask is not None:
            mask=mask.unsqueeze(0)

        batch_size=query.size(0)
        query,key,value=[model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2) for model, x in zip(self.linears, (query, key, value))]
        x,self.attn=attention(query,key,value,mask=mask,dropout=self.dropout)
        x=x.transpose(1,2).contiguous().view(batch_size,-1,self.head*self.d_k)
        return self.linears[-1](x)


#multiheadedAttention test block
head = 8
embedding_dim =512
dropout=0.2
query=key=value=pe_result
mask=Variable(torch.zeros(8,4,4))
mha =MultiHeadedAttention(head,embedding_dim,dropout)
mha_result=mha(query,key,value,mask)
print(mha_result)
print(mha_result.shape)
print("***********************multiheadedAttention test block")


#positionWiseFeedForward

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w1=nn.Linear(d_model,d_ff)
        self.w2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        return self.w2(self.dropout(F.relu(self.w1(x))))

d_model=512
d_ff=64
dropout=0.2
x=mha_result
ff=PositionwiseFeedForward(d_model,d_ff,dropout=dropout)
ff_result=ff(x)
print(ff_result)
print(ff_result.shape)
print("***********************positionWiseFeedForward test block")

#layer norm

class LayerNome(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNome,self).__init__()
        self.a2=nn.Parameter(torch.ones(features))
        self.b2=nn.Parameter(torch.zeros(features))
        self.eps=eps

    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        std =x.std(-1,keepdim=True)
        return self.a2*(x-mean)/(std+self.eps)+self.b2

features=512
ln=LayerNome(features=features)
lnn=nn.LayerNorm(features)
ln_result=ln(x)
lnn_result=lnn(x)
print(ln_result)
print(ln_result.shape)
print(lnn_result)
print(lnn_result.shape)
print("***********************layer norm test block")

class SublayerConnection(nn.Module):
    def __init__(self,size,dropout=0.1):
        super(SublayerConnection,self).__init__()
        self.norm=nn.LayerNorm(size)
        self.dropout=nn.Dropout(dropout)
        self.size=size

    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))
size = 512
dropout = 0.2
head = 8
d_model = 512
x = pe_result
mask = Variable(torch.zeros(8, 4, 4))
self_attn = MultiHeadedAttention(head, d_model)
sublayer = lambda x: self_attn(x, x, x, mask)
sc = SublayerConnection(size, dropout)
sc_result = sc(x, sublayer)
print(sc_result)
print(sc_result.shape)
print("***********************SublayerConnection test block")

#EncoderLayer
class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        super(EncoderLayer,self).__init__()
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        self.size=size
        self.sublayer=clone(SublayerConnection(size,dropout),2)


    def forward(self,x,mask):
        x=self.sublayer[0](x,lambda x:self_attn(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward)

size = 512
head = 8
d_model = 512
d_ff = 64
x = pe_result
dropout = 0.2
self_attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
mask = Variable(torch.zeros(8, 4, 4))
el = EncoderLayer(size, self_attn, ff, dropout)
el_result = el(x, mask)
print(el_result)
print(el_result.shape)
print("***********************EncoderLayer test block")

#Encoder
class Encoder(nn.Module):
    def __init__(self,layer,N):
        super(Encoder.self).__init__()
        self.layers=clone(layer,N)
        self.norm=nn.LayerNorm(layer.size)

    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)

            