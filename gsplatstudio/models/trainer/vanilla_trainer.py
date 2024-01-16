import torch
from random import randint
import gsplatstudio
from pathlib import Path
from gsplatstudio.models.trainer.base_trainer import BaseTrainer
from gsplatstudio.utils.type_utils import *
from gsplatstudio.utils.progress_bar import ProgressBar

@dataclass
class GaussTrainerConfig:
    detect_anomaly: bool = False
    iterations: int = 30000
    save_iterations: list = field(default_factory=list)
    test_iterations: list = field(default_factory=list)
    ckpt_iterations: list = field(default_factory=list)

@gsplatstudio.register("vanilla-trainer")
class VanillaTrainer(BaseTrainer):
    @property
    def config_class(self):
        return GaussTrainerConfig
    @property    
    def state(self):
        return {
            "data": self.data.spatial_scale,
            "model": self.model.state,
            "structOptim": self.structOptim.state,
            "paramOptim": self.paramOptim.state,
            "iteration": self.iteration
        }
    
    def setup_components(self):

        spatial_lr_scale = self.data.spatial_scale
        
        # init model from data
        self.model.init_from_pcd(self.data.point_cloud, spatial_lr_scale)
        
        # init paramOptim from model
        param_lr_group = self.model.create_param_lr_groups(self.paramOptim.cfg)
        self.paramOptim.init_optim(param_lr_group, spatial_lr_scale, self.cfg.iterations)
        
        # init structOptim from model
        self.structOptim.init_optim(self.model, spatial_lr_scale)

        # init progress bar
        self.progress_bar = ProgressBar(first_iter=0, total_iters=self.cfg.iterations)
    
    def restore_components(self, system_path, iteration):
        ckpt_path = Path(system_path) / f"{iteration}.pth"
        try:
            ckpt_dict = torch.load(ckpt_path)
            spatial_lr_scale = ckpt_dict["data"]
            
            pcd_path = Path(self.view_dir) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
            self.model.load_ply(str(pcd_path))
            self.model.restore(state = ckpt_dict["model"], spatial_lr_scale = spatial_lr_scale)
            
            self.structOptim.restore(state = ckpt_dict["structOptim"], spatial_lr_scale = spatial_lr_scale)
            
            param_lr_group = self.model.create_param_lr_groups(self.paramOptim.cfg)
            self.paramOptim.restore(state = ckpt_dict["paramOptim"], spatial_lr_scale = spatial_lr_scale, param_lr_group = param_lr_group, max_iter = self.cfg.iterations)
            
            self.first_iteration = ckpt_dict["iteration"] + 1
            # init progress bar
            self.progress_bar = ProgressBar(first_iter=self.first_iteration, total_iters=self.cfg.iterations)
            
        except Exception as e:
            self.logger.warning(f"Cannot load {ckpt_path}! Error: {e} Train from scratch")
            self.setup_components()

    def train(self) -> None:

        ema_loss_for_log = 0.0
        viewpoint_stack = None
        is_white_background = self.renderer.background_color == [255,255,255]
        
        for iteration in range(self.first_iteration, self.cfg.iterations + 1):    
            self.paramOptim.update_lr(iteration)
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.model.increment_sh_degree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = self.data.get_train_pair_list().copy()
            viewpoint_pair = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            # Render
            render_pkg = self.renderer.render(model = self.model, camera = viewpoint_pair.camera)

            # Loss
            gt_image = viewpoint_pair.image.get_resolution_data_from_path(self.data.cfg.resolution, self.data.cfg.resolution_scales[0])
            loss = self.loss(render_pkg["render"], gt_image)
            loss.backward()
            self.iteration = iteration

            with torch.no_grad():
                
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                self.progress_bar.update(iteration, ema_loss_for_log=ema_loss_for_log)
                self.recorder.snapshot("ema_loss_for_log", ema_loss_for_log)
                self.recorder.snapshot("loss", loss.clone().detach().cpu().item())

                # Log and save
                if iteration in self.cfg.save_iterations:
                    self.save_scene(iteration)

                # Densification
                self.structOptim.update_optim(iteration, self.model, self.paramOptim, render_pkg, is_white_background)
                
                # Optimizer step
                self.paramOptim.update_optim(iteration)

                # Recorder step
                self.recorder.update(iteration)

                # Checkpoint saving step
                if iteration in self.cfg.ckpt_iterations:
                    self.save_ckpt(iteration)

        

    






