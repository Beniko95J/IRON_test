import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import traceback
import trimesh
from copy import deepcopy


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

suffix = '.exr'
class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print("Load data: Begin")
        self.device = torch.device("cuda")
        self.conf = conf

        self.data_dir = conf.get_string("data_dir")
        self.render_cameras_name = conf.get_string("render_cameras_name")
        self.object_cameras_name = conf.get_string("object_cameras_name")

        self.camera_outside_sphere = conf.get_bool("camera_outside_sphere", default=True)
        # self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)  # not used

        # Scale_mat: transform the object to unit sphere for training
        self.load_sdf_grid = True
        if self.load_sdf_grid:
            sdf_grid_dict = np.load('datasets/sdf_grid_filled.npy', allow_pickle=True).item()
            sdf_grid_dict_corrected = np.load('datasets/sdf_grid_filled_corrected.npy', allow_pickle=True).item()
            self.sdf_grid = sdf_grid_dict_corrected['grid']
            self.sdf_grid = torch.from_numpy(self.sdf_grid).to(self.device)
            # self.sdf_grid = self.sdf_grid.permute(0, 2, 1)
            # self.sdf_grid = torch.flip(self.sdf_grid, dims=[1])
            center = sdf_grid_dict['center']
            scale = sdf_grid_dict['scale']
        else:
            pcd = trimesh.load('datasets/000.obj')
            vertices = pcd.vertices
            bbox_max = np.max(vertices, axis=0) 
            bbox_min = np.min(vertices, axis=0) 
            center = (bbox_max + bbox_min) * 0.5
            radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max() 
            scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
            scale_mat[:3, 3] = center

            # Scale_mat: transform the reconstructed mesh in unit sphere to original space with scale 150 for evaluation
            self.scale_mat = deepcopy(scale_mat)
            self.scale_mat[0, 0] *= 150
            self.scale_mat[1, 1] *= 150
            self.scale_mat[2, 2] *= 150
            self.scale_mat[:3, 3] *= 150
            # import pdb; pdb.set_trace()
            self.sdf_grid = None

        import json

        camera_dict = json.load(open(os.path.join(self.data_dir, "cam_dict_norm.json")))
        for x in list(camera_dict.keys()):
            x = x[:-4] + suffix
            camera_dict[x]["K"] = np.array(camera_dict[x]["K"]).reshape((4, 4))
            w2c = np.array(camera_dict[x]["W2C"]).reshape((4, 4))
            # OpenGL -> OpenCV
            # w2c = w2c @ np.diag([1., -1., -1., 1.])
            camera_dict[x]["W2C"] = w2c

        self.camera_dict = camera_dict

        try:
            self.images_lis = sorted(glob(os.path.join(self.data_dir, "image/*.png")))
            self.n_images = len(self.images_lis)
            self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 255.0
        except:
            # traceback.print_exc()

            print("Loading png images failed; try loading exr images")
            import pyexr

            self.images_lis = sorted(glob(os.path.join(self.data_dir, "image/*.exr")))
            self.n_images = len(self.images_lis)
            self.images_np = np.clip(
                np.power(np.stack([pyexr.open(im_name).get()[:, :, :-1][:, :, ::-1] for im_name in self.images_lis]), 1.0 / 2.2),
                0.0,
                1.0,
            )
            # import pdb; pdb.set_trace()

        no_mask = False
        if no_mask:
            print("Not using masks")
            self.masks_lis = None
            self.masks_np = np.ones_like(self.images_np)
        else:
            try:
                self.masks_lis = sorted(glob(os.path.join(self.data_dir, "mask/*.png")))
                self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 255.0
            except:
                # traceback.print_exc()

                print("Loading mask images failed; try not using masks")
                self.masks_lis = None
                self.masks_np = np.ones_like(self.images_np)

        self.images_np = self.images_np[..., :3]
        self.masks_np = self.masks_np[..., :3]

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [np.eye(4).astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        self.world_mats_np = []
        for x in self.images_lis:
            x = os.path.basename(x)[:-4] + suffix
            K = self.camera_dict[x]["K"].astype(np.float32)
            W2C = self.camera_dict[x]["W2C"].astype(np.float32)
            C2W = np.linalg.inv(self.camera_dict[x]["W2C"]).astype(np.float32)
            self.intrinsics_all.append(torch.from_numpy(K))
            self.pose_all.append(torch.from_numpy(C2W))
            self.world_mats_np.append(W2C)

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        print("image shape, mask shape: ", self.images.shape, self.masks.shape)
        print("image pixel range: ", self.images.min().item(), self.images.max().item())

        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)  # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        for i in range(self.pose_all.shape[0]):
            if self.load_sdf_grid:
                scale_mat = np.diag([scale, scale, scale, 1.0]).astype(np.float32)
                scale_mat = torch.from_numpy(scale_mat).cuda()
                self.pose_all[i, :3, 3:] = self.pose_all[i, :3, 3:] - torch.from_numpy(center).cuda()[..., None]
                self.pose_all[i, :, 3:] = scale_mat @ (self.pose_all[i, :, 3:])
                # import pdb; pdb.set_trace()
            else:
                self.pose_all[i, :, 3:] = torch.from_numpy(np.linalg.inv(scale_mat)).cuda() @ self.pose_all[i, :, 3:]

        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.00, -1.00, -1.00, 1.0])
        object_bbox_max = np.array([1.00, 1.00, 1.00, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.eye(4).astype(np.float32)
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print("Load data: End")
        # import pdb; pdb.set_trace()

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size, precrop_ratio=0.2):
        """
        Generate random rays at world space from one camera.
        """

        if precrop_ratio > 0:
            start_x, end_x = int(precrop_ratio * self.W), int((1- precrop_ratio) * self.W)
            pixels_x = torch.randint(low=start_x, high=end_x, size=[batch_size])
            pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        else:
            pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
            pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()  # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        if self.images_lis[idx].endswith(".exr"):
            import pyexr

            img = np.power(pyexr.open(self.images_lis[idx]).get()[:, :, :-1][:, :, ::-1], 1.0 / 2.2) * 255.0
        else:
            img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255).astype(np.uint8)
