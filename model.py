from __future__ import annotations
import torch, torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class VGGSpatialEncoder(nn.Module):
    def __init__(self, pretrained=True, trainable_last_blocks=True):
        super().__init__(); m=vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features=m.features
        if trainable_last_blocks:
            for p in self.features[:24].parameters(): p.requires_grad=False
        else:
            for p in self.features.parameters(): p.requires_grad=False
    def forward(self,x):
        f=self.features(x)               # B,512,7,7 for 224x224
        return f.permute(0,2,3,1).reshape(x.size(0),49,512)

class BiSpatialRNN(nn.Module):
    def __init__(self,in_dim,hidden=192,dropout=0.0):
        super().__init__(); self.rnn=nn.RNN(in_dim,hidden,batch_first=True,bidirectional=True,
                                            nonlinearity='tanh',dropout=0.0)
        self.out_dim=2*hidden
    def forward(self,x):
        y,_=self.rnn(x); return y.mean(dim=1)

class SpatialConvBaseline(nn.Module):
    def __init__(self,in_dim,out_dim=256):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(in_dim,out_dim,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        self.out_dim=out_dim
    def forward(self,tokens):
        b,l,d=tokens.shape; x=tokens.reshape(b,7,7,d).permute(0,3,1,2); return self.net(x).flatten(1)

class AAOMARN(nn.Module):
    def __init__(self, selected_idx=None, attention_dim=256, heads=4, rnn_hidden=192, dropout=0.3,
                 pretrained=True, use_mha=True, use_rnn=True, use_spatial2d=False):
        super().__init__(); self.encoder=VGGSpatialEncoder(pretrained)
        idx=torch.arange(512) if selected_idx is None else torch.as_tensor(selected_idx,dtype=torch.long)
        self.register_buffer('selected_idx',idx)
        self.proj=nn.Linear(len(idx),attention_dim)
        self.use_mha=use_mha; self.use_rnn=use_rnn; self.use_spatial2d=use_spatial2d
        self.mha=nn.MultiheadAttention(attention_dim,heads,batch_first=True,dropout=dropout) if use_mha else None
        if use_spatial2d:
            self.context=SpatialConvBaseline(attention_dim,attention_dim); outdim=attention_dim
        elif use_rnn:
            self.context=BiSpatialRNN(attention_dim,rnn_hidden); outdim=2*rnn_hidden
        else:
            self.context=None; outdim=attention_dim
        self.norm=nn.LayerNorm(attention_dim); self.drop=nn.Dropout(dropout); self.fc=nn.Linear(outdim,1)
    def forward(self,x,return_attention=False):
        t=self.encoder(x).index_select(-1,self.selected_idx); t=self.proj(t)
        att=None
        if self.mha is not None:
            a,att=self.mha(t,t,t,need_weights=return_attention,average_attn_weights=False); t=self.norm(t+a)
        if self.context is None: z=t.mean(1)
        else: z=self.context(t)
        logit=self.fc(self.drop(z)).squeeze(-1)
        return (logit,att) if return_attention else logit
