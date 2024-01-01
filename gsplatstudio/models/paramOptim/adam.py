import torch
import torch.nn as nn
import gsplatstudio
from gsplatstudio.utils.general_utils import get_expon_lr_func
from gsplatstudio.utils.config import parse_structured
from gsplatstudio.utils.type_utils import *

@dataclass
class adamAcustomlrConfig:
    position_lr_delay_mult: float = 0.01
    position_lr_final: float = 1.6e-06
    position_lr_init:float =  0.00016
    position_lr_max_steps: float = 30000
    feature_lr: float = 0.0025
    rotation_lr: float = 0.001
    scaling_lr: float = 0.005
    opacity_lr: float = 0.05


@gsplatstudio.register("adam+customLR-paramOptim")
class adamAcustomlr:
    def __init__(self, cfg):
        self.cfg = parse_structured(adamAcustomlrConfig, cfg)
        
    def init_optim(self,param_lr_group, spatial_lr_scale, max_iter):
        self.optimizer = torch.optim.Adam(param_lr_group, lr=0.0, eps=1e-15)
        self.xyz_lr_schedule = get_expon_lr_func(lr_init=self.cfg.position_lr_init*spatial_lr_scale,
                                                lr_final=self.cfg.position_lr_final*spatial_lr_scale,
                                                lr_delay_mult=self.cfg.position_lr_delay_mult,
                                                max_steps=self.cfg.position_lr_max_steps)
        self.max_iter = max_iter

    def update_lr(self,iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_lr_schedule(iteration)
                param_group['lr'] = lr
                return lr
            
    def update_optim(self,iteration):
        if iteration < self.max_iter:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = True)

    def prune_optim(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        
        return optimizable_tensors

    def replace_tensor(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def print_state(self):
        state_dict = self.optimizer.state_dict()
        for param_group in state_dict['param_groups']:
            name = param_group['name']
            idx = param_group['params'][0]
            exp_avg = state_dict['state'][idx]['exp_avg']
            exp_avg_sq = state_dict['state'][idx]['exp_avg_sq']
            print(f"{name}: exp_avg-{exp_avg.shape}, exp_avg_sq-{exp_avg_sq.shape}")
