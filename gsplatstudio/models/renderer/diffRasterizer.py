import math
import torch
import gsplatstudio
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gsplatstudio.utils.sh_utils import eval_sh
from gsplatstudio.utils.type_utils import *
from gsplatstudio.models.renderer.base_renderer import BaseRenderer

@dataclass
class DiffRasterizerRendererConfig:
    background_color: list = field(default_factory=list)
    scaling_modifier: float = 1.0
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False
    prefiltered: bool = False
    override_color: list = field(default_factory=list)
    use_full_opacity: bool = False
    

@gsplatstudio.register("diffRasterizer-renderer")
class DiffRasterizerRenderer(BaseRenderer):
    
    @property
    def config_class(self):
        return DiffRasterizerRendererConfig
    
    @property
    def background_color(self):
        if self.cfg.background_color == [-1,-1,-1]: # random background
            return torch.rand((3), device="cuda")
        else:
            return torch.tensor(self.cfg.background_color, dtype=torch.float32, device="cuda")

    def render(self, representation, camera):
        """
        Render the scene. 
        Background tensor (bg_color) must be on GPU!
        """
    
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(representation.xyz, dtype=representation.xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(camera.fov_x * 0.5)
        tanfovy = math.tan(camera.fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.height),
            image_width=int(camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.background_color,
            scale_modifier=self.cfg.scaling_modifier,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=representation.sh_degree,
            campos=camera.camera_center,
            prefiltered=self.cfg.prefiltered,
            debug=self.cfg.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = representation.xyz
        means2D = screenspace_points
        if self.cfg.use_full_opacity:
            opacity = torch.ones_like(representation.opacity, dtype=representation.opacity.dtype, requires_grad=False, device="cuda")
        else:
            opacity = representation.opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if self.cfg.compute_cov3D_python:
            cov3D_precomp = representation.covariance(self.cfg.scaling_modifier)
        else:
            scales = representation.scaling
            rotations = representation.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if self.cfg.override_color == [-1,-1,-1]:
            if self.cfg.convert_SHs_python:
                shs_view = representation.features.transpose(1, 2).view(-1, 3, (representation.max_sh_degree+1)**2)
                dir_pp = (representation.xyz - camera.camera_center.repeat(representation.features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(representation.sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = representation.features
        else:
            colors_precomp = self.cfg.override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}