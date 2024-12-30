import hydra
from geometry_perception_utils.io_utils import get_abs_path
from geometry_perception_utils.pinhole_utils import project_pp_depth_from_hfov
from geometry_perception_utils.image_utils import get_color_array
from geometry_perception_utils.vispy_utils import plot_color_plc
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous, eulerAnglesToRotationMatrix
from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
from imageio.v2 import imread
from geometry_perception_utils.dense_voxel_grid import VoxelGrid2D, VoxelGrid3D
import pickle


def project_and_register_rgbd(cfg, frame):
    # read RGB, Depth, Camera Pose files
    rgb = imread(f"{cfg.scene.rgb_dir}/{frame}.jpg")
    depth = np.load(f"{cfg.scene.depth_dir}/{frame}.npy")
    cam_pose = np.load(f"{cfg.scene.poses_dir}/{frame}.npy")

    # Project the depth to 3D points in Camera Coordinate (CC)
    xyz_cc, m = project_pp_depth_from_hfov(depth, cfg.scene.hfov)
    # Transform the 3D points from Camera Coordinate to World Coordinate (WC)
    xyz_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz_cc)
    # Get the color of the 3D points
    xyz_rgb = get_color_array(rgb)[:, m]/255
    return np.vstack((xyz_wc, xyz_rgb))


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    list_frames = [Path(f).stem for f in os.listdir(cfg.scene.rgb_dir)]

    # Loading pre-computed Voxels grid from saved pickle bin files
    bins_2d = pickle.load(Path(cfg.voxel_grid_2d.bins_fn).open('rb'))
    bins_3d = pickle.load(Path(cfg.voxel_grid_3d.bins_fn).open('rb'))
    voxel2d = VoxelGrid2D.from_bins(*bins_2d)
    voxel3d = VoxelGrid3D.from_bins(*bins_3d)

    global_xyz_rgb = []
    for frame in tqdm(list_frames):
        # Read RGB, Depth, Camera Pose and register them to WC
        xyz_rgb_wc = project_and_register_rgbd(cfg, frame)

        # This function project xyz_wc (only 3D points) into a grid map (2D or 3D)
        xyz_wc_vx, vx_idx, xyz_idx, all_xyz_idx = voxel3d.project_xyz(
            xyz_rgb_wc[:3])

        # xyz_wc_vx: Voxels centers
        # xyz_idx: indexes that maps wc -> vx domains
        local_xyz = np.vstack([xyz_wc_vx, xyz_rgb_wc[3:, xyz_idx]])
        global_xyz_rgb.append(local_xyz)

    xyz_rgb_wc = np.hstack(global_xyz_rgb)
    plot_color_plc(xyz_rgb_wc[:3, :].T, xyz_rgb_wc[3:, :].T)


if __name__ == "__main__":
    main()
