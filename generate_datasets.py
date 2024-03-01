import os
import json
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
import OpenEXR
import Imath
import array
import trimesh
import shutil
import json
from collections import OrderedDict


intrinsic = np.array([[1.050000000000000000e+03, 0.000000000000000000e+00, 5.400000000000000000e+02, 0.0],
                              [0.000000000000000000e+00, 1.050000000000000000e+03, 3.600000000000000000e+02, 0.0],
                              [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00, 0.0],
                              [0.0, 0.0, 0.0, 1.0]])


def load_exr(exr_path):
    img = OpenEXR.InputFile(exr_path)
    dw = img.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    img_R, img_G, img_B, img_A = img.channels('RGBA', pt)

    arr_R = array.array('f', img_R)
    arr_G = array.array('f', img_G)
    arr_B = array.array('f', img_B)
    arr_A = array.array('f', img_A)

    ndarr_R = np.array(arr_R, dtype="float32")
    ndarr_G = np.array(arr_G, dtype="float32")
    ndarr_B = np.array(arr_B, dtype="float32")
    ndarr_A = np.array(arr_A, dtype="float32")

    ndarr_RGBA = np.array([[r, g, b, a] for r, g, b, a in zip(ndarr_R, ndarr_G, ndarr_B, ndarr_A)], dtype="float32")
    ndarr_RGBA = ndarr_RGBA.reshape(size[1], size[0], 4) # (H, W, 4) FIXME: maximum value is not 1.0!

    return ndarr_RGBA


holdout = 8
dst_root = 'datasets/synthetic_face_sparse/holdout{:02d}'.format(holdout)
dst_train_dir = os.path.join(dst_root, 'train')
dst_test_dir = os.path.join(dst_root, 'test')
dst_train_imgdir = os.path.join(dst_train_dir, 'image')
dst_train_maskdir = os.path.join(dst_train_dir, 'mask')
dst_test_imgdir = os.path.join(dst_test_dir, 'image')
dst_test_maskdir = os.path.join(dst_test_dir, 'mask')
os.makedirs(dst_train_imgdir, exist_ok=True)
os.makedirs(dst_train_maskdir, exist_ok=True)
os.makedirs(dst_test_imgdir, exist_ok=True)
os.makedirs(dst_test_maskdir, exist_ok=True)

train_idx = [n for n in range(35) if n % holdout == 0]
test_idx = [n for n in range(35) if n % holdout != 0]

# Train
cam_dict = OrderedDict()
src_root = 'datasets/synthetic_face'
for i in train_idx:
    src_pair = os.path.join(src_root, 'pair_{}'.format(i))
    exr_path = os.path.join(src_pair, 'img_view0_c.exr')
    mask_np = load_exr(exr_path)[..., 3:4] # (H, W, 1)
    mask_np = np.repeat(mask_np, 3, axis=-1) # (H, W, 3)
    mask_np = mask_np.astype(np.uint8) * 255

    shutil.copy(exr_path, os.path.join(dst_train_imgdir, '{:02d}.exr'.format(i)))
    cv.imwrite(os.path.join(dst_train_maskdir, '{:02d}.png'.format(i)), mask_np)

    w2c_view0 = np.loadtxt(os.path.join(src_pair, 'w2c_view0.txt'))
    cam_dict['{:02d}.exr'.format(i)] = {'K': intrinsic.flatten().tolist(), 'W2C': w2c_view0.flatten().tolist(), 'img_size': np.array((1080, 720)).tolist()}

with open(os.path.join(dst_train_dir, 'cam_dict_norm.json'), 'w') as f:
    json.dump(cam_dict, f, indent=4)

# Test
cam_dict = OrderedDict()
src_root = 'datasets/synthetic_face'
for i in test_idx:
    src_pair = os.path.join(src_root, 'pair_{}'.format(i))
    exr_path = os.path.join(src_pair, 'img_view0_c.exr')
    mask_np = load_exr(exr_path)[..., 3:4] # (H, W, 1)
    mask_np = np.repeat(mask_np, 3, axis=-1) # (H, W, 3)
    mask_np = mask_np.astype(np.uint8) * 255

    shutil.copy(exr_path, os.path.join(dst_test_imgdir, '{:02d}.exr'.format(i)))
    cv.imwrite(os.path.join(dst_test_maskdir, '{:02d}.png'.format(i)), mask_np)

    w2c_view0 = np.loadtxt(os.path.join(src_pair, 'w2c_view0.txt'))
    cam_dict['{:02d}.exr'.format(i)] = {'K': intrinsic.flatten().tolist(), 'W2C': w2c_view0.flatten().tolist(), 'img_size': np.array((1080, 720)).tolist()}

with open(os.path.join(dst_test_dir, 'cam_dict_norm.json'), 'w') as f:
    json.dump(cam_dict, f, indent=4)
