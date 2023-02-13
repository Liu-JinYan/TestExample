import  torch
from  torch import  nn
from torch.utils.data import  DataLoader
import  torchvision
from  torchvision import  datasets
import  numpy as np
import  sys
import  os

def to_pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViT(nn.Module):
    def __init__(
            self, image_size, patch_size,
            num_classes=1000, dim=1024, depth=6, num_heads=8, mlp_dim=2048,
            pool='cls', channels=3, dim_per_head=64, dropout=0., embed_dropout=0.
    ):
        super().__init__()

        img_h, img_w = to_pair(image_size)
        self.patch_h, self.patch_w = to_pair(patch_size)
        assert not img_h % self.patch_h and not img_w % self.patch_w, \
            f'Image dimensions ({img_h},{img_w}) must be divisible by the patch size ({self.patch_h},{self.patch_w}).'
        num_patches = (img_h // self.patch_h) * (img_w // self.patch_w)

        assert pool in {'cls', 'mean'}, f'pool type must be either cls (cls token) or mean (mean pooling), got: {pool}'

        patch_dim = channels * self.patch_h * self.patch_w
        self.patch_embed = nn.Linear(patch_dim, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # Add 1 for cls_token
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.dropout = nn.Dropout(p=embed_dropout)

        self.transformer = Transformer(
            dim, mlp_dim, depth=depth, num_heads=num_heads,
            dim_per_head=dim_per_head, dropout=dropout
        )

        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        b, c, img_h, img_w = x.shape
        assert not img_h % self.patch_h and not img_w % self.patch_w, \
            f'Input image dimensions ({img_h},{img_w}) must be divisible by the patch size ({self.patch_h},{self.patch_w}).'

        '''i. Patch partition'''
        num_patches = (img_h // self.patch_h) * (img_w // self.patch_w)
        # (b,c,h,w)->(b,n_patches,patch_h*patch_w*c)
        patches = x.view(
            b, c,
            img_h // self.patch_h, self.patch_h,
            img_w // self.patch_w, self.patch_w
        ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1)

        '''ii. Patch embedding'''
        # (b,n_patches,dim)
        tokens = self.patch_embed(patches)
        # (b,n_patches+1,dim)
        tokens = torch.cat([self.cls_token.repeat(b, 1, 1), tokens], dim=1)
        tokens += self.pos_embed[:, :(num_patches + 1)]
        tokens = self.dropout(tokens)

        '''iii. Transformer Encoding'''
        enc_tokens = self.transformer(tokens)

        '''iv. Pooling'''
        # (b,dim)
        pooled = enc_tokens[:, 0] if self.pool == 'cls' else enc_tokens.mean(dim=1)

        '''v. Classification'''
        # (b,n_classes)
        logits = self.mlp_head(pooled)

        return logits

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
        self.decoder= Transformer(
           decoder_dim,
           decoder_dim*4,
           depth=decoder_depth,
           num_heads= num_decoder_heads,
            dim_per_head=decoder_dim_per_head
        )

        self.decoder_pos_embed = nn.Embedding(num_patchs_plus_cls_token -1 , decoder_dim)
        self.head = nn.Linear( decoder_dim,num_pixel_per_patch)

class PreNorm(nn.Module):
    def __init__(self,dim,net):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = net
    def forward(self,x,**kwargs):
        return  self.net(self.norm(x),**kwargs)
class SelfAttintion(nn.Module):
    def __init__(self,dim,num_heads=8,dim_per_head=64,dropout=0.):
        super().__init__()

        self.num_heads = num_heads
        self.scale = dim_per_head ** -0.5

        inner_dim = dim_per_head * num_heads
        self.to_qkv  = nn.Linear(dim,inner_dim*3,bias=False)

        self.attend = nn.Softmax(dim=-1)

        project_out = not (num_heads ==1 and dim_per_head == dim)
        self.out = nn.Sequential(
            nn.Linear(inner_dim,dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self,x):
        b,l,d=x.shape

        qkv=self.to_qkv(x)
        qkv = qkv.view(b,l,3,self.num_heads,-1).permute(2,0,3,1,4).contiguous()
        q,k,v=qkv.chunk(3)
        q,k,v =  q.squeeze(0),k.sequeeze(0),v.sequuze(0)

        attn = self.attend(
            torch.matmul(q,k.transpoes(-1,-2)) * self.scale
        )

        z= torch.matmul(attn,v)
        z= z.transpose(1,2).reshape(b,l,-1)

        out =self.out(z)

        return  out


class FFN(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)

        )
    def forward(self,x):
        return  self.net(x)



class Transformer(nn.Module):
    def __init__(self,dim,mlp_dim,depth=6,num_heads=6,dim_per_head=64,dropout=0.):
        super().__init__()

        self.layers=nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim,SelfAttintion(dim,num_heads=num_heads,dim_per_head=dim_per_head,dropout=dropout)),
                    PreNorm(dim,FFN(dim,mlp_dim,dropout))
                ])
            )
    def forward(self,x):
        for norm_attn,norm_ffn in self.layers:
            x += norm_attn(x)
            x += norm_ffn(x)
        return  x





device = torch.device('mps')

# 读入图像并缩放到适合模型输入的尺寸
from PIL import Image

BASE_DIR='/Users/liujinyan/'


img_raw = Image.open(os.path.join(BASE_DIR, 'test.jpg'))
h, w = img_raw.height, img_raw.width
ratio = h / w
print(f"image hxw: {h} x {w} mode: {img_raw.mode}")

img_size, patch_size = (224, 224), (16, 16)
img = img_raw.resize(img_size)
rh, rw = img.height, img.width
print(f'resized image hxw: {rh} x {rw} mode: {img.mode}')
img.save(os.path.join( BASE_DIR,'test1.jpg'))

# 将图像转换成张量
from torchvision.transforms import ToTensor, ToPILImage

img_ts = ToTensor()(img).unsqueeze(0).to(device)
print(f"input tensor shape: {img_ts.shape} dtype: {img_ts.dtype} device: {img_ts.device}")

# 实例化模型并加载训练好的权重
encoder = ViT(img_size, patch_size, dim=512, mlp_dim=1024, dim_per_head=64)
decoder_dim = 512
mae = MAE(encoder, decoder_dim, decoder_depth=6)
weight = torch.load(os.path.join(BASE_DIR, 'mae.pth'), map_location='cpu')
mae.to(device)

# 推理
# 模型重建的效果图，mask 效果图

recons_img_ts, masked_img_ts = predict(img_ts)
recons_img_ts, masked_img_ts = recons_img_ts.cpu().squeeze(0), masked_img_ts.cpu().squeeze(0)

# 将结果保存下来以便和原图比较
recons_img = ToPILImage()(recons_img_ts)
recons_img.save(os.path.join(BASE_DIR, 'recons_mountain.jpg'))

masked_img = ToPILImage()(masked_img_ts)
masked_img.save(os.path.join(BASE_DIR, 'masked_mountain.jpg'))

@torch.no_grad
def predict(self, x):
    self.eval()

    device = x.device
    b, c, h, w = x.shape

    '''i. Patch partition'''

    num_patches = (h // self.patch_h) * (w // self.patch_w)
    # (b, c=3, h, w)->(b, n_patches, patch_size**2*c)
    patches = x.view(
        b, c,
        h // self.patch_h, self.patch_h,
        w // self.patch_w, self.patch_w
    ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1)

    '''ii. Divide into masked & un-masked groups'''

    num_masked = int(self.mask_ratio * num_patches)

    # Shuffle
    # (b, n_patches)
    shuffle_indices = torch.rand(b, num_patches, device=device).argsort()
    mask_ind, unmask_ind = shuffle_indices[:, :num_masked], shuffle_indices[:, num_masked:]

    # (b, 1)
    batch_ind = torch.arange(b, device=device).unsqueeze(-1)
    mask_patches, unmask_patches = patches[batch_ind, mask_ind], patches[batch_ind, unmask_ind]

    '''iii. Encode'''

    unmask_tokens = self.encoder.patch_embed(unmask_patches)
    # Add position embeddings
    unmask_tokens += self.encoder.pos_embed.repeat(b, 1, 1)[batch_ind, unmask_ind + 1]
    encoded_tokens = self.encoder.transformer(unmask_tokens)

    '''iv. Decode'''

    enc_to_dec_tokens = self.enc_to_dec(encoded_tokens)

    # (decoder_dim)->(b, n_masked, decoder_dim)
    mask_tokens = self.mask_embed[None, None, :].repeat(b, num_masked, 1)
    # Add position embeddings
    mask_tokens += self.decoder_pos_embed(mask_ind)

    # (b, n_patches, decoder_dim)
    concat_tokens = torch.cat([mask_tokens, enc_to_dec_tokens], dim=1)
    # dec_input_tokens = concat_tokens
    dec_input_tokens = torch.empty_like(concat_tokens, device=device)
    # Un-shuffle
    dec_input_tokens[batch_ind, shuffle_indices] = concat_tokens
    decoded_tokens = self.decoder(dec_input_tokens)

    '''v. Mask pixel Prediction'''

    dec_mask_tokens = decoded_tokens[batch_ind, mask_ind, :]
    # (b, n_masked, n_pixels_per_patch=patch_size**2 x c)
    pred_mask_pixel_values = self.head(dec_mask_tokens)

    # 比较下预测值和真实值
    mse_per_patch = (pred_mask_pixel_values - mask_patches).abs().mean(dim=-1)
    mse_all_patches = mse_per_patch.mean()

    print(
        f'mse per (masked)patch: {mse_per_patch} mse all (masked)patches: {mse_all_patches} total {num_masked} masked patches')
    print(f'all close: {torch.allclose(pred_mask_pixel_values, mask_patches, rtol=1e-1, atol=1e-1)}')

    '''vi. Reconstruction'''

    recons_patches = patches.detach()
    # Un-shuffle (b, n_patches, patch_size**2 * c)
    recons_patches[batch_ind, mask_ind] = pred_mask_pixel_values
    # 模型重建的效果图
    # Reshape back to image
    # (b, n_patches, patch_size**2 * c)->(b, c, h, w)
    recons_img = recons_patches.view(
        b, h // self.patch_h, w // self.patch_w,
        self.patch_h, self.patch_w, c
    ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

    mask_patches = torch.randn_like(mask_patches, device=mask_patches.device)
    # mask 效果图
    patches[batch_ind, mask_ind] = mask_patches
    patches_to_img = patches.view(
        b, h // self.patch_h, w // self.patch_w,
        self.patch_h, self.patch_w, c
    ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

    return recons_img, patches_to_img
