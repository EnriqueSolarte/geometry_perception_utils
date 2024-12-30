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


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):

    # List of all frames
    list_frame = [fn.stem for fn in Path(cfg.scene.rgb_dir).iterdir()]
    all_xyz_rgb_wc = []
    for frame in tqdm(list_frame[1:-1:2], desc="Processing frames"):
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
        
        all_xyz_rgb_wc.append(np.vstack((xyz_wc, xyz_rgb)))
        # Visualization of each frame in 3D
        plot_color_plc(xyz_wc.T, xyz_rgb.T)

    # Concatenate all the frames
    xyz_rgb_wc = np.hstack(all_xyz_rgb_wc)
    # Visualization of all frames in 3D
    plot_color_plc(xyz_rgb_wc[:3, :].T, xyz_rgb_wc[3:, :].T)
    

if __name__ == "__main__":
    main()
