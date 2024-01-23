import torch
import gsplatstudio
from gsplatstudio.utils.gaussian_utils import build_rotation, inverse_sigmoid
from gsplatstudio.utils.type_utils import *
from gsplatstudio.models.structOptim.base_structOptim import BaseStructOptim

@dataclass
class SplitWithCloneWithPruneConfig:
    max_sh_drgree: int = 3
    percent_dense: float = 0.01
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densify_grad_threshold: float = 0.0002
    densification_interval: int = 100
    min_opacity: float = 0.005
    num_split: int = 2
    size_threshold: int = 20

@gsplatstudio.register("split+clone+prune-structOptim")
class SplitWithCloneWithPrune(BaseStructOptim):
    
    @property
    def config_class(self):
        return SplitWithCloneWithPruneConfig

    @property
    def state(self):
        return (
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom  
        )
    
    def _restore(self, state, spatial_lr_scale):
        (self.max_radii2D,
        self.xyz_gradient_accum,
        self.denom) = state
        self.spatial_lr_scale = spatial_lr_scale

    def init_optim(self, representation, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale
        self.reset_stats(representation)      
        
    def update_optim(self, iteration, representation, paramOptim, render_pkg, is_white_background):
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        if iteration < self.cfg.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
            self.xyz_gradient_accum[visibility_filter] += torch.norm(viewspace_point_tensor.grad[visibility_filter,:2], dim=-1, keepdim=True)
            self.denom[visibility_filter] += 1

            if iteration > self.cfg.densify_from_iter and iteration % self.cfg.densification_interval == 0:
                self.densify_and_prune(iteration, representation, paramOptim)
            if iteration % self.cfg.opacity_reset_interval == 0 or (is_white_background and iteration == self.cfg.densify_from_iter):
                self.reset_model_opacity(representation, paramOptim)
    
    def should_start_limit_size(self,iteration):
        return iteration > self.cfg.opacity_reset_interval

    def densify_and_prune(self, iteration, representation, paramOptim):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        self.densify_and_clone(representation, paramOptim, grads)
        self.densify_and_split(representation, paramOptim, grads)

        prune_mask = (representation.opacity < self.cfg.min_opacity).squeeze()
        if self.should_start_limit_size(iteration):
            big_points_vs = self.max_radii2D > self.cfg.size_threshold
            big_points_ws = representation.scaling.max(dim=1).values > 0.1 * self.spatial_lr_scale
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask, representation, paramOptim)
        torch.cuda.empty_cache()

    def densify_and_clone(self, representation, paramOptim, grads):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= self.cfg.densify_grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(representation.scaling, dim=1).values <= self.cfg.percent_dense*self.spatial_lr_scale)
        
        new_tensors_dict = {
            "xyz": representation._xyz[selected_pts_mask],
            "f_dc": representation._features_dc[selected_pts_mask],
            "f_rest": representation._features_rest[selected_pts_mask],
            "opacity": representation._opacity[selected_pts_mask],
            "scaling" : representation._scaling[selected_pts_mask],
            "rotation" : representation._rotation[selected_pts_mask]
        }
        
        self.densification_postfix(representation, paramOptim, new_tensors_dict)

    def densify_and_split(self, representation, paramOptim, grads):
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((representation.xyz.shape[0]), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= self.cfg.densify_grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(representation.scaling, dim=1).values > self.cfg.percent_dense*self.spatial_lr_scale)

        stds = representation.scaling[selected_pts_mask].repeat(self.cfg.num_split,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(representation._rotation[selected_pts_mask]).repeat(self.cfg.num_split,1,1)
        
        new_tensors_dict = {
            "xyz": torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + representation.xyz[selected_pts_mask].repeat(self.cfg.num_split, 1),
            "f_dc": representation._features_dc[selected_pts_mask].repeat(self.cfg.num_split,1,1),
            "f_rest": representation._features_rest[selected_pts_mask].repeat(self.cfg.num_split,1,1),
            "opacity": representation._opacity[selected_pts_mask].repeat(self.cfg.num_split,1),
            "scaling" : representation.scaling_inverse_activation(representation.scaling[selected_pts_mask].repeat(self.cfg.num_split,1) / (0.8*self.cfg.num_split)),
            "rotation" : representation._rotation[selected_pts_mask].repeat(self.cfg.num_split,1)
        }

        self.densification_postfix(representation, paramOptim, new_tensors_dict)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(self.cfg.num_split * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, representation, paramOptim)

    def prune_points(self, mask, representation, paramOptim):
        valid_points_mask = ~mask
        optimizable_tensors = paramOptim.prune_optim(valid_points_mask)
        representation.update_params(optimizable_tensors)

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def densification_postfix(self, representation, paramOptim, new_tensors_dict):
        optimizable_tensors = paramOptim.cat_tensors(new_tensors_dict)
        representation.update_params(optimizable_tensors)
        self.reset_stats(representation)

    def reset_model_opacity(self, representation, paramOptim):
        opacities_new = inverse_sigmoid(torch.min(representation.opacity, torch.ones_like(representation.opacity)*0.01))
        optimizable_tensors = paramOptim.replace_tensor(opacities_new, "opacity")
        representation._opacity = optimizable_tensors["opacity"]

    def reset_stats(self, representation):
        self.xyz_gradient_accum = torch.zeros((representation.xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((representation.xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((representation.xyz.shape[0]), device="cuda")

