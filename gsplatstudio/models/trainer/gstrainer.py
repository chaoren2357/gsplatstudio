import torch
from pathlib import Path
from random import randint
import gsplatstudio
from gsplatstudio.utils.progress_bar import ProgressBar
from gsplatstudio.utils.type_utils import *
from gsplatstudio.utils.config import parse_structured

@dataclass
class GaussTrainerConfig:
    detect_anomaly: bool = False
    iterations: int = 30000
    start_checkpoint: list = field(default_factory=list)
    save_iterations: list = field(default_factory=list)
    test_iterations: list = field(default_factory=list)
    ckpt_iterations: list = field(default_factory=list)

@gsplatstudio.register("gaussian-trainer")
class GaussTrainer:
    def __init__(self, cfg) -> None:
        self.cfg = parse_structured(GaussTrainerConfig, cfg)

    def load(self, logger, data, model, loss, structOptim, paramOptim, renderer, checkpoint=None):
        self.logger = logger
        self.data = data
        self.model = model
        self.loss = loss
        self.structOptim = structOptim
        self.paramOptim = paramOptim
        self.renderer = renderer
        self.checkpoint = checkpoint

        spatial_lr_scale = self.data.spatial_scale
        
        # init model from data
        self.model.init_from_pcd(self.data.scene_info.point_cloud, spatial_lr_scale)
        
        # init paramOptim from model
        param_lr_group = self.model.create_param_lr_groups(self.paramOptim.cfg)
        self.paramOptim.init_optim(param_lr_group, spatial_lr_scale, self.cfg.iterations)
        
        # init structOptim from model
        self.structOptim.init_optim(self.model, spatial_lr_scale)
        
        # init progress bar
        self.progress_bar = ProgressBar(first_iter=0, total_iters=self.cfg.iterations)

    def train(self) -> None:
        
        first_iter = 1
        # TODO: Add checkpoint restore
        # if self.cfg.checkpoint:
        #     (model_params, first_iter) = torch.load(checkpoint)
        #     self.model.restore(model_params, opt)
        ema_loss_for_log = 0.0
        viewpoint_stack = None
        is_white_background = self.renderer.background_color == [255,255,255]
        
        for iteration in range(first_iter, self.cfg.iterations + 1):    
            self.paramOptim.update_lr(iteration)
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.model.increment_sh_degree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = self.data.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            # Render
            render_pkg = self.renderer.render(model = self.model, camera = viewpoint_cam)
            image = render_pkg["render"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            loss = self.loss(image, gt_image)
            loss.backward()
            # print(f"{iteration}, rotation step d, {self.model._rotation.grad}")
            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                self.progress_bar.update(iteration, ema_loss_for_log=ema_loss_for_log)

                # Log and save
                if iteration in self.cfg.save_iterations:
                    self.logger.info(f"\n[ITER {iteration}] Saving Gaussians")
                    self.save(iteration)

                # Densification
                self.structOptim.update(iteration, self.model, self.paramOptim, render_pkg, is_white_background)
                
                # Optimizer step
                self.paramOptim.update_optim(iteration)
                
                # Checkpoint saving step
                # if iteration in self.cfg.ckpt_iterations:
                #     self.logger.info(f"\n[ITER {iteration}] Saving Checkpoint")
                #     torch.save((self.model.capture(), iteration), self.data.trial_dir + "/chkpnt" + str(iteration) + ".pth")

    def save(self, iteration):
        ply_path = Path(self.data.trial_dir) / f"point_cloud/iteration_{iteration}" / "point_cloud.ply"
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_ply(ply_path)




