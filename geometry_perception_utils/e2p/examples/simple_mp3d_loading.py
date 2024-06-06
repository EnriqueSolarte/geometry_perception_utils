import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from vslab_360_datasets import MP3D
from vslab_360_datasets.dataloader.images_dataloader import ImageDataloader
from ly_camera_projection.e2p.e2p import e2p
from geometry_perception_utils.config_utils import read_cfg
import os
from functools import partial
import argparse
from imageio import imwrite


def example_using_list_ly(args):
    # ! Loading MP3D dataset
    datasets_fn = os.path.join(DIR_LY_CAM_PROJ_CFG, 'datasets.yaml')
    cfg = read_cfg(datasets_fn)
    mp3d = MP3D(cfg.mp3d)

    # ! Reading list_ly
    list_ly = mp3d.get_list_ly()

    # ! Setting output function to Simple dataloader
    partial_e2p = partial(
        e2p,
        u_deg=0,
        v_deg=0,
        fov_deg=args.fov_deg,
        out_hw=args.out_hw,
        in_rot_deg=0,
        mode='bilinear',
    )

    # ! Loop iteration
    for ly in tqdm(list_ly):
        image_360 = ly.get_rgb()
        image_pp = partial_e2p(image_360)

        fn = os.path.join(DIR_LY_CAM_PROJ_ASSETS, f'image_pp.jpg')
        imwrite(fn, image_pp)
        fn = os.path.join(DIR_LY_CAM_PROJ_ASSETS, f'image_360.jpg')
        imwrite(fn, image_360)

        break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--fov_deg", type=float, default=90)
    parser.add_argument("--out_hw", type=int, nargs=2, default=(512, 512))
    args = parser.parse_args()

    example_using_list_ly(args)
