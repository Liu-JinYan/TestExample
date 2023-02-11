import  torch
from  torch import  nn
from torch.utils.data import  DataLoader
import  torchvision
from  torchvision import  datasets
import  numpy as np

class MAE(nn.Module):
    def __init__(self,encoder,decoder_dim,mask_ratio=0.75,decoder_depth=1,num_decoder_heads=8,decoder_dim_per_head=64):
        super().__init__()
        assert 0. < mask_ratio<1.,f'mask ratio must be kept between 0 and 1, got: {mask_ratio}'

        #Encoder
        self.encoder=encoder
        self.patch_h,self.patch_w=encoder.patch_h,encoder.patch_w

        num_patchs_plus_cls_token,encoder_dim = encoder.pos_embed.shape[-2:]
        num_pixel_per_patch = encoder.pos_embed.weight.size(1)

        self.enc_to_dec = nn.Linear( encoder_dim,decoder_dim) if encoder_dim != decoder_dim else nn.Identity()

        #mask  token

        self.mask_ratio =mask_ratio
        self.mask_embed = nn.parameter(torch.randn(decoder_dim))

        #decoder
        self.decoder= Transformen()


class PreNorm(nn.Module):
    def __init__(self,dim,net):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = net
    def forward(self,x,**kwargs):
        return  self.net(self.norm(x),**kwargs)
class SelfAttintion(nn.Module):
    


class Transformer(nn.Module):
    def __init__(self,dim,mlp_dim,depth=6,num_head=6,dim_per_head=64,dropout=0.):
        super().__init__()

        self.layers=nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim,)
                ])
            )
