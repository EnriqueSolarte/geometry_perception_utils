import hydra
from geometry_perception_utils.io_utils import get_abs_path
from pathlib import Path
from tqdm import tqdm
import logging
from imageio.v2 import imread
import numpy as np
from geometry_perception_utils.pinhole_utils import project_pp_depth_from_hfov
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous
from geometry_perception_utils.image_utils import get_color_array
from geometry_perception_utils.vispy_utils import plot_color_plc
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

    # List of all frames
    list_frame = [fn.stem for fn in Path(cfg.scene.rgb_dir).iterdir()]
    voxel2d = VoxelGrid2D(cfg.voxel_grid_2d)
    voxel3d = VoxelGrid3D(cfg.voxel_grid_3d)

    for frame in tqdm(list_frame, desc="Processing frames"):
        xyz_rgb_wc = project_and_register_rgbd(cfg, frame)
        # Creating voxels grid
        _ = voxel2d.project_xyz(xyz_rgb_wc[:3, :])
        _ = voxel3d.project_xyz(xyz_rgb_wc[:3, :])

    # saving voxel grid maps as pickle
    fn = f"{cfg.voxel_grid_2d.bins_fn}"
    pickle.dump(voxel2d.get_bins(), open(fn, "wb"))
    logging.info(f"2D voxel map saved @ {fn}")
    fn = f"{cfg.voxel_grid_3d.bins_fn}"
    pickle.dump(voxel3d.get_bins(), open(fn, "wb"))
    logging.info(f"3D voxel map saved @ {fn}")


if __name__ == "__main__":
    main()
