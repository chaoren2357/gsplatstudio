import torch
import gsplatstudio
from gsplatstudio.utils.general_utils import build_rotation, inverse_sigmoid
from gsplatstudio.utils.type_utils import *
from gsplatstudio.utils.config import parse_structured

@dataclass
class splitAcloneApruneConfig:
    max_sh_drgree: int = 3
    percent_dense: float = 0.01
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densify_grad_threshold: float = 0.0002
    densification_interval: int = 100
    size_threshold: int = 20
    min_opacity: float = 0.005
    num_split: int = 2

@gsplatstudio.register("split.clone.prune-structOptim")
class splitAcloneAprune:
    def __init__(self, cfg):
        self.cfg = parse_structured(splitAcloneApruneConfig, cfg)

    @property
    def state(self):
        return (
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom  
        )
    
    def restore(self, state, spatial_lr_scale):
        (self.max_radii2D,
        self.xyz_gradient_accum,
        self.denom) = state
        self.spatial_lr_scale = spatial_lr_scale

    def init_optim(self,model, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale
        self.reset_stats(model)      
        
    def update(self, iteration, model, paramOptim, render_pkg, is_white_background):
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        if iteration < self.cfg.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
            self.xyz_gradient_accum[visibility_filter] += torch.norm(viewspace_point_tensor.grad[visibility_filter,:2], dim=-1, keepdim=True)
            self.denom[visibility_filter] += 1

            if iteration > self.cfg.densify_from_iter and iteration % self.cfg.densification_interval == 0:
                self.densify_and_prune(iteration, model, paramOptim)
            if iteration % self.cfg.opacity_reset_interval == 0 or (is_white_background and iteration == self.cfg.densify_from_iter):
                self.reset_model_opacity(model, paramOptim)
    
    def should_start_limit_size(self,iteration):
        return iteration > self.cfg.opacity_reset_interval

    def densify_and_prune(self, iteration, model, paramOptim):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        self.densify_and_clone(model, paramOptim, grads)
        self.densify_and_split(model, paramOptim, grads)

        prune_mask = (model.opacity < self.cfg.min_opacity).squeeze()
        if self.should_start_limit_size(iteration):
            big_points_vs = self.max_radii2D > self.cfg.size_threshold
            big_points_ws = model.scaling.max(dim=1).values > 0.1 * self.spatial_lr_scale
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask, model, paramOptim)
        torch.cuda.empty_cache()

    def densify_and_clone(self, model, paramOptim, grads):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= self.cfg.densify_grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(model.scaling, dim=1).values <= self.cfg.percent_dense*self.spatial_lr_scale)
        
        new_tensors_dict = {
            "xyz": model._xyz[selected_pts_mask],
            "f_dc": model._features_dc[selected_pts_mask],
            "f_rest": model._features_rest[selected_pts_mask],
            "opacity": model._opacity[selected_pts_mask],
            "scaling" : model._scaling[selected_pts_mask],
            "rotation" : model._rotation[selected_pts_mask]
        }
        
        self.densification_postfix(model, paramOptim, new_tensors_dict)

    def densify_and_split(self, model, paramOptim, grads):
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((model.xyz.shape[0]), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= self.cfg.densify_grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(model.scaling, dim=1).values > self.cfg.percent_dense*self.spatial_lr_scale)

        stds = model.scaling[selected_pts_mask].repeat(self.cfg.num_split,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(model._rotation[selected_pts_mask]).repeat(self.cfg.num_split,1,1)
        
        new_tensors_dict = {
            "xyz": torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + model.xyz[selected_pts_mask].repeat(self.cfg.num_split, 1),
            "f_dc": model._features_dc[selected_pts_mask].repeat(self.cfg.num_split,1,1),
            "f_rest": model._features_rest[selected_pts_mask].repeat(self.cfg.num_split,1,1),
            "opacity": model._opacity[selected_pts_mask].repeat(self.cfg.num_split,1),
            "scaling" : model.scaling_inverse_activation(model.scaling[selected_pts_mask].repeat(self.cfg.num_split,1) / (0.8*self.cfg.num_split)),
            "rotation" : model._rotation[selected_pts_mask].repeat(self.cfg.num_split,1)
        }

        self.densification_postfix(model, paramOptim, new_tensors_dict)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(self.cfg.num_split * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, model, paramOptim)

    def prune_points(self, mask, model, paramOptim):
        valid_points_mask = ~mask
        optimizable_tensors = paramOptim.prune_optim(valid_points_mask)
        model.update_params(optimizable_tensors)

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def densification_postfix(self, model, paramOptim, new_tensors_dict):
        optimizable_tensors = paramOptim.cat_tensors(new_tensors_dict)
        model.update_params(optimizable_tensors)
        self.reset_stats(model)

    def reset_model_opacity(self, model, paramOptim):
        opacities_new = inverse_sigmoid(torch.min(model.opacity, torch.ones_like(model.opacity)*0.01))
        optimizable_tensors = paramOptim.replace_tensor(opacities_new, "opacity")
        model._opacity = optimizable_tensors["opacity"]

    def reset_stats(self, model):
        self.xyz_gradient_accum = torch.zeros((model.xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((model.xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((model.xyz.shape[0]), device="cuda")

