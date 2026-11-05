"""Frozen AnyPPG/PaPaGei temporal interfaces for dense PPG evaluation.

These preserve native preprocessing and time-varying hidden states. They never
turn a pooled embedding into a waveform. The PaPaGei taps and decoder capacity
are fixed before dense-task test results are examined.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.ppg_beat_models import AnyPPGEventHead, anyppg_input, frozen_stages, STAGE_CHANNELS
from scripts.ppg_model_expansion_pilot import _upstream, _state_hash, _weights

PAPAGEI_TAPS=(0,3,7,11,15,17)
PAPAGEI_CHANNELS=(32,32,64,128,256,512)


@torch.no_grad()
def papagei_stages(wrapper,x):
    if wrapper.training or any(p.requires_grad for p in wrapper.parameters()):
        raise ValueError('Temporal encoder must be frozen and in eval mode')
    encoder=wrapper.encoder
    out=encoder.first_block_conv(x)
    if encoder.use_bn:
        out=encoder.first_block_bn(out)
    out=encoder.first_block_relu(out)
    outputs=[]
    for index,block in enumerate(encoder.basicblock_list):
        out=block(out)
        if index in PAPAGEI_TAPS:
            outputs.append(out)
    if tuple(y.shape[1] for y in outputs)!=PAPAGEI_CHANNELS:
        raise ValueError('PaPaGei temporal channels differ from the fixed adapter')
    return outputs


class PaPaGeiEventHead(nn.Module):
    """27,367 parameters versus AnyPPG's 26,737; same decoder and output clock."""
    def __init__(self):
        super().__init__()
        self.projections=nn.ModuleList([nn.Sequential(nn.Conv1d(c,11,1),nn.GroupNorm(1,11),nn.ReLU()) for c in PAPAGEI_CHANNELS])
        self.decoder=nn.Sequential(nn.Conv1d(66,32,5,padding=2),nn.GroupNorm(8,32),nn.ReLU(),
                                   nn.Conv1d(32,32,5,padding=4,dilation=2),nn.GroupNorm(8,32),nn.ReLU(),nn.Conv1d(32,1,1))

    def forward(self,stages):
        if len(stages)!=6 or tuple(x.shape[1] for x in stages)!=PAPAGEI_CHANNELS:
            raise ValueError('Expected the six registered PaPaGei temporal maps')
        projected=[F.interpolate(layer(x),size=512,mode='linear',align_corners=False) for layer,x in zip(self.projections,stages)]
        return self.decoder(torch.cat(projected,dim=1)).squeeze(1)


def make_head(method):
    if method in ('AnyPPG','AnyPPG-random-frozen'):
        return AnyPPGEventHead()
    if method in ('PaPaGei-S','PaPaGei-P'):
        return PaPaGeiEventHead()
    raise ValueError(method)


class TemporalEncoder:
    def __init__(self,method,seed=42,device='cpu'):
        random=method=='AnyPPG-random-frozen'
        base='AnyPPG' if random else method
        if base not in ('AnyPPG','PaPaGei-S','PaPaGei-P'):
            raise ValueError('Unsupported temporal model')
        configs,revision=_upstream()
        from ppg_bench.models import get_port,external_module
        cfg=configs[base]
        self.port=get_port(cfg)
        torch.manual_seed(seed)
        if random:
            from ppg_bench.encoders.anyppg import ANYPPG_CFG
            source=external_module(cfg['init']['source_root'],'load_anyppg.resnet1d')
            self.encoder=source.Net1D(**ANYPPG_CFG)
        else:
            self.encoder=self.port.load_encoder()
        self.encoder.float().to(device).eval().requires_grad_(False)
        self.method,self.base,self.device,self.config=method,base,device,cfg
        self.before=_state_hash(self.encoder)
        self.info={'method':method,'random_seed':seed if random else None,'upstream_commit':revision,
                   'checkpoint_sha256':{} if random else _weights(cfg),'state_before_sha256':self.before,
                   'encoder_parameters':sum(p.numel() for p in self.encoder.parameters()),'encoder_trainable_parameters':0,
                   'input_fs':125,'input_samples':1000,'output_fs':64,'output_samples':512,
                   'taps':'all six AnyPPG stage_list outputs' if base=='AnyPPG' else list(PAPAGEI_TAPS),
                   'channels':list(STAGE_CHANNELS if base=='AnyPPG' else PAPAGEI_CHANNELS),
                   'preprocessing':'AnyPPG original onset PoC Butterworth 0.5-8Hz and NumPy per-window zscore' if base=='AnyPPG' else 'Pinned official PaPaGei 0.5-12Hz Chebyshev II plus smoothing and per-window zscore',
                   'device':device,'precision':'FP32; no autocast'}

    @torch.no_grad()
    def __call__(self,windows125):
        x=np.asarray(windows125,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=1000 or not np.isfinite(x).all():
            raise ValueError('Temporal input requires finite 8-second windows at 125Hz')
        if self.base=='AnyPPG':
            prepared=torch.from_numpy(anyppg_input(x)).unsqueeze(1).to(self.device)
            outputs=frozen_stages(self.encoder,prepared)
        else:
            prepared=self.port.prepare_batch({'x':torch.from_numpy(x)},self.config,self.device)
            outputs=papagei_stages(self.encoder,prepared)
        if not all(torch.isfinite(z).all() for z in outputs):
            raise ValueError('Nonfinite temporal features')
        return outputs

    def audit(self):
        after=_state_hash(self.encoder)
        if after!=self.before or self.encoder.training or any(p.requires_grad for p in self.encoder.parameters()):
            raise ValueError('Frozen temporal encoder changed')
        return {**self.info,'state_after_sha256':after}
