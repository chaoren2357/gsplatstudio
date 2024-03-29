import torch
import numpy as np
from PIL import Image

import torch
from torch import nn
import torchvision.transforms as transforms
from gsplatstudio.utils.graphics_utils import fov2focal, getWorld2View, getProjectionMatrix, rotmat2qvec

class BasicCamera(nn.Module):
    def __init__(self, R, T, fov_x, fov_y, height, width, 
                 z_far = 100.0, z_near = 0.01, uid = 0, device = 'cuda',
                 **kwargs):
        super(BasicCamera, self).__init__()
        self.R = R
        self.T = T
        self.fov_x, self.fov_y = fov_x, fov_y
        self.width, self.height = width, height
        self.z_far = z_far
        self.z_near = z_near
        self.uid = uid
        self.device = device

        for key, value in kwargs.items():
            setattr(self, key, value)
        self.trans=np.array([0.0, 0.0, 0.0]), 
        self.scale=1.0
    
    @property
    def world_view_transform(self):
        return torch.tensor(getWorld2View(self.R, self.T, self.trans, self.scale)).transpose(0, 1).to(self.device)
    @property
    def projection_matrix(self):
        return getProjectionMatrix(znear=self.z_near, zfar=self.z_far, fovX=self.fov_x, fovY=self.fov_y).transpose(0,1).to(self.device)
    @property
    def full_proj_transform(self):
        return (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0).to(self.device)
    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3].to(self.device)

class BasicImage:
    def __init__(self, data=None, device = 'cuda', path = None, name = None, gt_alpha_mask = None, keep_data = False, **kwargs):
        # data is a [channels, height, width] tensor
        self.device = device
        self.data = self.format_data(data, gt_alpha_mask)
        self.gt_alpha_mask = gt_alpha_mask
        self.channels, self.height, self.width = self.data.shape
        self.path, self.name = path,name
        for key, value in kwargs.items():
            setattr(self, key, value)
        if not keep_data:
            self.data = None
        self.resolution_data_dict = {}
        
    @staticmethod
    def format_data(data, gt_alpha_mask):
        if data is None:
            return None
        # Convert data to a PyTorch tensor
        if isinstance(data, Image.Image):
            np_image = np.array(data)
            if np_image.shape[2] == 4 and gt_alpha_mask is None:
                gt_alpha_mask = np_image[:, :, 3]
                gt_alpha_mask = torch.from_numpy(gt_alpha_mask) / 255.0
                np_image = np_image[:, :, :3]
            image = torch.from_numpy(np_image) / 255.0
            data = image.permute(2, 0, 1)
        elif isinstance(data, np.ndarray):
            if data.shape[2] == 4 and gt_alpha_mask is None:
                gt_alpha_mask = np_image[:, :, 3]
                gt_alpha_mask = torch.from_numpy(gt_alpha_mask) / 255.0
                np_image = np_image[:, :, :3]
            data = torch.from_numpy(data) / 255.0
        elif isinstance(data, torch.Tensor):
            data = data.clone().detach()
            if data.max() > 1.0:
                data = data.float() / 255
        else:
            print(f"Image data should be in [Image.Image, np.ndarray, torch.Tensor], but get {type(data)}")
        
        # Multiply gt_alpha_mask
        if gt_alpha_mask is not None:
            data *= gt_alpha_mask

        return data
    
    def to_device(self, device):
        self.device = device
        self.data = self.data.to(device)

    def set(self,**kwargs):
        if 'data' in kwargs:
            self.data = self.format_data(kwargs['data'], self.gt_alpha_mask)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_data_from_path(self):
        data = Image.open(self.path)
        data = self.format_data(data, self.gt_alpha_mask)
        data = data.to(self.device)
        return data

    def get_resolution(self, resolution_input,resolution_scale):
        orig_w, orig_h = self.width, self.height
        # resolution in (height, width)
        if resolution_input in [1, 2, 4, 8]:
            resolution = round(orig_h/(resolution_scale * resolution_input)), round(orig_w/(resolution_scale * resolution_input))
        else:  # should be a type that converts to float
            if resolution_input == -1:
                if orig_w > 1600:
                    global_down = orig_w / 1600
                else:
                    global_down = 1
            else:
                global_down = orig_w / resolution_input

            scale = float(global_down) * float(resolution_scale)
            resolution = (int(orig_h / scale), int(orig_w / scale))
        return resolution

    def get_resolution_data_from_path(self, resolution_input, resolution_scale):
        resolution = self.get_resolution(resolution_input, resolution_scale)
        if self.resolution_data_dict.get(resolution) is None:
            resize_transform = transforms.Resize(resolution)
            data = Image.open(self.path)
            data = self.format_data(data, self.gt_alpha_mask)
            resized_image_tensor = resize_transform(data.unsqueeze(0)).squeeze(0)
            data = resized_image_tensor.to(self.device)
            self.resolution_data_dict[resolution] = data
        data = self.resolution_data_dict.get(resolution)
        return data

class CameraImagePair:
    def __init__(self, cam: BasicCamera, img: BasicImage, uid: int, **kwargs):
        self.camera = cam
        self.image = img
        self.uid = uid
        for key, value in kwargs.items():
            setattr(self, key, value)
    @property
    def json(self):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = self.camera.R.transpose()
        Rt[:3, 3] = self.camera.T
        Rt[3, 3] = 1.0

        W2C = np.linalg.inv(Rt)
        pos = W2C[:3, 3]
        rot = W2C[:3, :3]
        serializable_array_2d = [x.tolist() for x in rot]
        return  {
            'id' : self.uid,
            'img_name' : self.image.name,
            'width' : self.camera.width,
            'height' : self.camera.height,
            'position': pos.tolist(),
            'rotation': serializable_array_2d,
            'fy' : fov2focal(self.camera.fov_y, self.camera.height),
            'fx' : fov2focal(self.camera.fov_x, self.camera.width)
        }


def transform_camera_from_carla_matrix_to_colmap_W2C_quaternion(camera_data):
    x_carla,y_carla,z_carla,roll_carla,pitch_carla,yaw_carla = camera_data['x'],camera_data['y'],camera_data['z'],camera_data['roll'],camera_data['pitch'],camera_data['yaw']
    x = y_carla
    y = -z_carla
    z = x_carla
    roll = pitch_carla
    pitch = yaw_carla
    yaw = roll_carla
    C2W_matrix = get_transform_matrix(x, y, z, pitch, roll, yaw)
    W2C_matrix = np.linalg.inv(C2W_matrix)
    W2C_quaternion = rotmat2qvec(W2C_matrix[:3, :3])
    W2C_translation = W2C_matrix[:3, 3]
    return W2C_quaternion, W2C_translation

def get_transform_matrix(x, y, z, pitch, roll, yaw):
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    
    transform = np.array([
        [ cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr,  x],
        [ sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr,  y],
        [-sp     ,  cp * sr,                 cp * cr,                 z],
        [0., 0., 0., 1.]
    ])
    return transform

def transform_camera_from_carla_matrix_to_colmap_C2W(camera_data):
    x_carla,y_carla,z_carla,roll_carla,pitch_carla,yaw_carla = camera_data['x'],camera_data['y'],camera_data['z'],camera_data['roll'],camera_data['pitch'],camera_data['yaw']
    x = y_carla
    y = -z_carla
    z = x_carla
    roll = pitch_carla
    pitch = yaw_carla
    yaw = roll_carla
    C2W_matrix = get_transform_matrix(x, y, z, pitch, roll, yaw)
    return np.array(C2W_matrix)

def transform_camera_from_carla_matrix_to_colmap_W2C(camera_data):
    x_carla,y_carla,z_carla,roll_carla,pitch_carla,yaw_carla = camera_data['x'],camera_data['y'],camera_data['z'],camera_data['roll'],camera_data['pitch'],camera_data['yaw']
    x = y_carla
    y = -z_carla
    z = x_carla
    roll = pitch_carla
    pitch = yaw_carla
    yaw = roll_carla
    C2W_matrix = get_transform_matrix(x, y, z, pitch, roll, yaw)
    W2C_matrix = np.linalg.inv(C2W_matrix)
    return np.array(W2C_matrix)

def transform_camera_from_matrixcity_matrix_to_colmap_W2C_quaternion(camera_data):
    c2w = np.array(camera_data['transform_matrix'])
    c2w[:3, :3] *= 100
    c2w[0:3, 2] *= -1
    c2w[0:3, 1] *= -1
    w2c = np.linalg.inv(c2w)
    w2c_quaternion = rotmat2qvec(w2c[:3, :3])
    W2C_translation = w2c[:3, 3]
    return w2c_quaternion, W2C_translation

def transform_camera_from_matrixcity_matrix_to_colmap_W2C(camera_data):
    c2w = np.array(camera_data['transform_matrix'])
    c2w[:3, :3] *= 100
    c2w[0:3, 2] *= -1
    c2w[0:3, 1] *= -1
    w2c = np.linalg.inv(c2w)
    return w2c

def fov_to_focal_length(fov_degrees, width):
    fov_radians = np.radians(fov_degrees)
    focal_length = (width / 2) / np.tan(fov_radians / 2)
    return focal_length