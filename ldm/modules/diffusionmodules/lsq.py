from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np 
import tqdm
import math
from functools import partial

class RoundQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):    
        return input.round()
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n_lv, size):
        ctx.save_for_backward(torch.Tensor([n_lv, size]))
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        saved, = ctx.saved_tensors
        n_lv, size = int(saved[0]), float(saved[1])

        if n_lv == 0:
            return grad_output, None, None
        else:
            scale = 1 / np.sqrt(n_lv * size)
            return grad_output.mul(scale), None, None


class TDQ_Module(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = int(os.environ.get('TIME_EMBED_DIM'))
        dropout_ratio = float(os.environ.get('DROPOUT', 0.2))
        
        hidden_dim = 64
        self.s_gen = nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout_ratio),
                            nn.Linear(hidden_dim, 1),
                            nn.Softplus()
                    )
    
    def initialize(self, bb):
        self.s_gen[-2].bias.data.fill_(torch.log(torch.exp(bb) - 1.))

    def forward(self, time_emb):
        return self.s_gen(time_emb)


class _QuantAct(nn.Module):
    def __init__(self, manual_lv=None, dropout_p=0.0):
        super(_QuantAct, self).__init__()
        
        self.register_buffer('n_lv', torch.tensor(0))
        self.register_buffer('dynamic', torch.BoolTensor([False]))
        
        self.s          = Parameter(torch.Tensor(1)) # step size
        self.b          = Parameter(torch.Tensor(1)) # zero point for Asym
        self.manual_lv  = manual_lv
        self.dropout_p  = dropout_p
        
        self.tdq_module = TDQ_Module()
        
        self.non_negative = False
        self.tensors = {"mean":[], "std": [], "min": [], "max": [], "abs_max": []}
        
        self.cnt = 0
    
    def gather(self, x):
        x = x.reshape(x.shape[0], -1)
        
        self.tensors["mean"].append(x.mean(dim=1))
        self.tensors["std"].append(x.std(dim=1))
        self.tensors["min"].append(x.min(dim=1).values)
        self.tensors["max"].append(x.max(dim=1).values)
    
    def initialize(self, n_lv, dynamic):
        self.dynamic.data.fill_(dynamic)
        
        if self.manual_lv != None :
            self.n_lv.data.fill_(self.manual_lv)
        else :
            self.n_lv.data.fill_(n_lv)
        
        self.tensors['mean'] = torch.cat(self.tensors['mean'], dim=0)
        self.tensors['std']  = torch.cat(self.tensors['std'],  dim=0)
        self.tensors['min']  = torch.cat(self.tensors['min'],  dim=0)
        self.tensors['max']  = torch.cat(self.tensors['max'],  dim=0)
        
        real_mean = self.tensors['mean'].mean()
        abs_mean  = self.tensors['max'].abs().mean()
        sigma     = self.tensors['std'].mean()
        
        # Initailize zero point
        self.b.data.fill_(real_mean)
        
        # Initialize step size
        if self.non_negative:
            init_s = (real_mean + 16 * sigma) / (self.n_lv - 1)
        else:
            init_s = (real_mean + 3 * sigma) / (self.n_lv//2 - 1)
        
        self.s.data.fill_(init_s)
        self.tdq_module.initialize(init_s)
        
        del self.tensors
    
    def act_quant(self, x, s):
        raise NotImplementedError("Q_Act - You should be define act_quant.")
        
    def forward(self, x):
        if self.n_lv == 0:
            return x

        if not self.dynamic: # LSQ
            s = self.s
            # s = GradientScale.apply(self.s, self.n_lv, x.numel() // x.shape[0])
            # s = s/(self.n_lv//2 - 1)
        else:
            sgen_input = self.global_buffer.GLOBAL_TIME_EMBED

            s = self.tdq_module(sgen_input)
            
            if len(x.shape) == 2:
                s = s.view(-1,1)
            elif len(x.shape) == 3:
                s = s.view(-1,1,1)
            elif len(x.shape) == 4:
                s = s.view(-1,1,1,1)
        
        x = self.act_quant(x, s)
        return x


class Q_Sym(_QuantAct):

    def act_quant(self, x, s):
        x = F.hardtanh(x / s,- (self.n_lv // 2 - 1), self.n_lv // 2 - 1)
        x = RoundQuant.apply(x) * s
        
        return x


class Q_ASym(_QuantAct):
    
    def act_quant(self, x, s):
        b = self.b
        x = F.hardtanh((x - b) / s,- (self.n_lv // 2 - 1), self.n_lv // 2 - 1)
        x = RoundQuant.apply(x) * s + b
        
        return x

class Q_SiLU(_QuantAct):
    
    def __init__(self, manual_lv=None, dropout_p=0.0):
        super().__init__(manual_lv, dropout_p)
        self.non_negative = True
    
    def act_quant(self, x, s):
        
        act_min = -0.2785 # SiLU's minimum value is -0.2785
        if self.training:
            act_min = act_min / (1 - self.dropout_p)
                
        x = F.hardtanh( (x - act_min) / s, 0, self.n_lv - 1) # Non-negative
        x = RoundQuant.apply(x) * s + act_min
        
        return x
    
class _QuantLayer(nn.Module):
    def __init__(self, *args, manual_lv=None, act_func=None, w=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        
        # NOTE : Please define fwd_func
        self.fwd_func = None
        
        self.register_buffer('n_lv', torch.tensor(0))
        self.s = Parameter(torch.Tensor(1))
        self.manual_lv = manual_lv
        self.dropout_p = w
        
        if act_func is None:
            self.act_func = QuantOps.Sym()
        
        elif act_func == nn.SiLU or act_func == 'SiLU':
            self.act_func = QuantOps.SiLU(dropout_p=w)
        
        else:
            raise NotImplementedError(f"Not supported {type(act_func)}")

    def initialize(self, n_lv):
        if self.manual_lv != None :
            self.n_lv.data.fill_(self.manual_lv)
        else :
            self.n_lv.data.fill_(n_lv)

        self.s.data.fill_(self.weight.data.max() / (self.n_lv//2 - 1))
    
    def _weight_quant(self):
        s = GradientScale.apply(self.s, self.n_lv, self.weight.numel())
        weight = F.hardtanh(self.weight / s,- (self.n_lv // 2 - 1), self.n_lv // 2 - 1)
        weight = RoundQuant.apply(weight) * s
        return weight
    
    def _act_quant(self, x):
        return self.act_func(x)
    
    def forward(self, x):
        assert self.fwd_func is not None
        
        if self.act_func is not None:
            x = self._act_quant(x)

        if self.n_lv == 0:
            return self.fwd_func(input=x, weight=self.weight, bias=self.bias)
        else:
            weight = self._weight_quant()
            return self.fwd_func(input=x, weight=weight, bias=self.bias)


class Q_Linear(_QuantLayer, nn.Linear):
    def __init__(self, *args, manual_lv=None, act_func=None, w=0.0, **kwargs):
        super().__init__(*args, manual_lv=manual_lv, act_func=act_func, w=w, **kwargs)
        
        self.fwd_func = F.linear

class Q_Conv1d(_QuantLayer, nn.Conv1d):
    def __init__(self, *args, manual_lv=None, act_func=None, w=0.0, **kwargs):
        super().__init__(*args, manual_lv=manual_lv, act_func=act_func, w=w, **kwargs)

        self.fwd_func = partial(F.conv1d, stride=self.stride, padding=self.padding, 
                                dilation=self.dilation, groups=self.groups)

class Q_Conv2d(_QuantLayer, nn.Conv2d):
    def __init__(self, *args, manual_lv=None, act_func=None, w=0.0, **kwargs):
        super().__init__(*args, manual_lv=manual_lv, act_func=act_func, w=w, **kwargs)
        
        self.fwd_func = partial(F.conv2d, stride=self.stride, padding=self.padding, 
                                dilation=self.dilation, groups=self.groups)


def initialize(model, loader, weight_n_lv, act_n_lv, dynamic, act=False, weight=False):
    print("WARNING : you have to call Q.initialize() AFTER .load_state_dict() !")
    model.quant_initialized = False
    
    weight_initialized = False
    def initialize_hook(module, input, output):
        if isinstance(module, _QuantAct) and act:
            module.gather(input[0].data)
            # module.initialize(act_n_lv, input[0].data)

        if isinstance(module, _QuantLayer) and weight and weight_initialized:
            module.initialize(weight_n_lv)

    hooks = []
    for name, module in model.named_modules():
        hook = module.register_forward_hook(initialize_hook)
        hooks.append(hook)

    # model.eval()
    # model.train() #SHIT
    device = model.device
    for i, batch in enumerate(loader):
        with torch.no_grad():
            batch[model.first_stage_key] = batch[model.first_stage_key].to(device)
            
            for t in tqdm.tqdm(range(100)):
                model.shared_step(batch)
                
            weight_initialized = True
            model.shared_step(batch)
        break
    
    for name, module in model.named_modules():
        if isinstance(module, _QuantAct) and act:
            module.initialize(act_n_lv, dynamic)
        
    # model.train()
    for hook in hooks:
        hook.remove()
        
    model.quant_initialized = True
    print("Initialization for Quant is Done..")


class QuantOps(object):
    initialize = initialize
    Sym = Q_Sym if os.environ.get('ASYM') is None else Q_ASym
    SiLU = Q_SiLU
    Conv1d = Q_Conv1d
    Conv2d = Q_Conv2d
    Linear = Q_Linear
