
from geometry_perception_utils.e2p import pinhole_mask_on_equirectangular
from geometry_perception_utils.io_utils import create_directory
from imageio import imwrite
from pathlib import Path

if __name__ == "__main__":
    """
    fov = 90 # FOV of the perspective camera
    u_deg=0 # yaw angle of the perspective camera 
    v_deg=0 # pitch angle of the perspective camera
    out_hw = (512, 1024) # output equirectangular image size
    """
    idx = 0
    vis_dir = create_directory(f"{Path(__file__).parent.__str__()}/vis")

    for theta in range(-180, 180, 45):
        for phi in range(-45, 45, 10):

            img = pinhole_mask_on_equirectangular(
                u_deg=theta, v_deg=phi, out_hw=(512, 1024))
            fn = f"{vis_dir}/pinhole_mask_on_equirectangular_{idx}.png"
            imwrite(f'{fn}', img)
            idx += 1
