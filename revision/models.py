"""Dispersion-guided classifier and explicit encoder alternatives."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from physics import delays_in_samples


class DispersionBank(nn.Module):
    def __init__(self, frequencies_ghz, dm_values, sampling_ms, time):
        super().__init__()
        delays = delays_in_samples(frequencies_ghz, dm_values, sampling_ms)
        if abs(delays).max() >= time:
            raise ValueError('DM-grid displacement exceeds window support; use adequate windows or a residual grid.')
        self.guard = math.ceil(abs(delays).max()) + time
        self.register_buffer('delays', torch.tensor(delays, dtype=torch.float32))
        self.register_buffer('dm_values', torch.tensor(dm_values, dtype=torch.float32))

    def forward(self, x):
        # Perform FFT in FP32: non-power-of-two FP16 FFT is not supported by CUDA.
        time = x.shape[-2]
        with torch.autocast(device_type=x.device.type, enabled=False):
            padded = F.pad(x.float(), (0,0,self.guard,self.guard))
            length = padded.shape[-2]
            spectrum = torch.fft.rfft(padded, dim=2)
            cycles = torch.fft.rfftfreq(length, device=x.device)
            bank = []
            for delay in self.delays:
                phase = torch.exp(2j*math.pi*cycles[:,None]*delay[None,:])
                aligned = torch.fft.irfft(spectrum*phase[None,None,:,:], n=length, dim=2)
                bank.append(aligned[:,:,self.guard:self.guard+time,:])
            return torch.stack(bank, dim=1)


class DMAttention(nn.Module):
    def __init__(self, channels, hypotheses):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((32,32))
        self.features = nn.Sequential(nn.Flatten(), nn.Linear(channels*1024,128), nn.ReLU(), nn.Linear(128,32), nn.ReLU())
        self.weights = nn.Sequential(nn.Linear(hypotheses*32,256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256,64), nn.ReLU(), nn.Linear(64,hypotheses))

    def forward(self, bank):
        b,k,c,t,f = bank.shape
        features = self.features(self.pool(bank.reshape(b*k,c,t,f))).reshape(b,k*32)
        weights = self.weights(features).softmax(dim=-1)
        return (bank*weights[:,:,None,None,None]).sum(dim=1)


class ConformerBlock(nn.Module):
    def __init__(self, dim, heads, kernel, dropout):
        super().__init__()
        def feedforward():
            return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim,dim*4), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim*4,dim), nn.Dropout(dropout))
        self.ff1, self.ff2 = feedforward(), feedforward()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True)
        self.conv_norm = nn.LayerNorm(dim)
        self.conv = nn.Sequential(nn.Conv1d(dim,2*dim,1), nn.GLU(dim=1), nn.Conv1d(dim,dim,kernel,padding=kernel//2,groups=dim), nn.BatchNorm1d(dim), nn.SiLU(), nn.Conv1d(dim,dim,1), nn.Dropout(dropout))
        self.out = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5*self.ff1(x)
        norm = self.attn_norm(x)
        x = x + self.attn(norm,norm,norm,need_weights=False)[0]
        x = x + self.conv(self.conv_norm(x).transpose(1,2)).transpose(1,2)
        return self.out(x+0.5*self.ff2(x))


class ConformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c,t,f = cfg['input_shape']
        dim = cfg['embedding_dim']
        layers=[]
        for out, stride in [(32,2),(64,2),(128,2),(256,4)]:
            layers += [nn.Conv2d(c,out,3,stride=(1,stride),padding=1),nn.BatchNorm2d(out),nn.ReLU()]
            c=out; f=math.ceil(f/stride)
        self.stem=nn.Sequential(*layers)
        self.projection=nn.Linear(c*f,dim)
        self.position=nn.Parameter(torch.randn(1,t,dim)*0.02)
        self.blocks=nn.Sequential(*[ConformerBlock(dim,cfg['attention_heads'],cfg['convolution_kernel'],cfg['dropout']) for _ in range(cfg['conformer_layers'])])
        self.head=nn.Sequential(nn.Linear(dim,128),nn.ReLU(),nn.Dropout(cfg['dropout']),nn.Linear(128,2))

    def forward(self,x):
        x=self.stem(x).permute(0,2,1,3).flatten(2)
        return self.head(self.blocks(self.projection(x)+self.position).mean(dim=1))


class ViTSmall(nn.Module):
    def __init__(self, shape):
        super().__init__()
        c,t,f=shape
        self.padding=(-f)%16,(-t)%16
        tokens=math.ceil(t/16)*math.ceil(f/16)
        self.patch=nn.Conv2d(c,384,16,stride=16)
        self.cls=nn.Parameter(torch.randn(1,1,384)*0.02)
        self.position=nn.Parameter(torch.randn(1,tokens+1,384)*0.02)
        layer=nn.TransformerEncoderLayer(384,6,1536,dropout=0.1,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,12,enable_nested_tensor=False)
        self.head=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,2))

    def forward(self,x):
        x=F.pad(x,(0,self.padding[0],0,self.padding[1]))
        tokens=self.patch(x).flatten(2).transpose(1,2)
        x=torch.cat((self.cls.expand(x.shape[0],-1,-1),tokens),dim=1)+self.position
        return self.head(self.encoder(x)[:,0])


def cnn_encoder(name, channels):
    from torchvision import models
    constructors = {'resnet18':models.resnet18,'efficientnet_b0':models.efficientnet_b0,'convnext_tiny':models.convnext_tiny,'mobilenet_v3_large':models.mobilenet_v3_large}
    if name not in constructors:
        raise ValueError(f'Unknown encoder: {name}')
    model=constructors[name](weights=None,num_classes=2)
    # Replace only the first convolution; all weights are initialized from scratch.
    for path,module in model.named_modules():
        if isinstance(module,nn.Conv2d):
            parent_path,_,leaf=path.rpartition('.')
            parent=model.get_submodule(parent_path) if parent_path else model
            parent._modules[leaf]=nn.Conv2d(channels,module.out_channels,module.kernel_size,module.stride,module.padding,bias=module.bias is not None)
            break
    return model


class Classifier(nn.Module):
    def __init__(self,cfg,encoder='conformer',front_end='learned'):
        super().__init__()
        if front_end not in {'raw','uniform','learned'}:
            raise ValueError('front_end must be raw, uniform, or learned.')
        self.shape=tuple(cfg['input_shape']); self.front_end=front_end
        if front_end!='raw':
            if cfg.get('frequencies_ghz') is None or cfg.get('dm_values') is None:
                raise ValueError('Fill actual frequencies_ghz and dm_values in the configuration.')
            if len(cfg['frequencies_ghz'])!=self.shape[-1]:
                raise ValueError('Frequency metadata must match the input frequency axis.')
            self.bank=DispersionBank(cfg['frequencies_ghz'],cfg['dm_values'],cfg['sampling_ms'],self.shape[-2])
            if front_end=='learned': self.fusion=DMAttention(self.shape[0],len(cfg['dm_values']))
        if encoder=='conformer': self.encoder=ConformerEncoder(cfg)
        elif encoder=='vit_small': self.encoder=ViTSmall(self.shape)
        else: self.encoder=cnn_encoder(encoder,self.shape[0])

    def forward(self,x):
        if tuple(x.shape[1:])!=self.shape:
            raise ValueError(f'Expected (*,{self.shape}), got {tuple(x.shape)}.')
        if self.front_end!='raw':
            bank=self.bank(x)
            x=self.fusion(bank) if self.front_end=='learned' else bank.mean(dim=1)
        return self.encoder(x)
