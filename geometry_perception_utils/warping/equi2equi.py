
# This lib is a wrapper of equilib
from equilib import Equi2Equi
import numpy as np
import torch
from geometry_perception_utils.warping.utils import preprocess, postprocess
from imageio.v2 import imwrite, imread


def main():
    img_equi_fn = '/media/Pluto/kike/geometry_perception_utils/geometry_perception_utils/data/equi2.png'
    results_dir = '/media/Pluto/kike/geometry_perception_utils/geometry_perception_utils/data/results'
    # Rotation:
    rot = {
        "roll": 0,  #
        "pitch": np.pi / 4,  # vertical
        "yaw": 0,  # horizontal
    }

    # Initialize equi2equi
    equi2equi = Equi2Equi()
    device = torch.device("cuda")

    # Open Image
    src_img = imread(img_equi_fn)[:, :, :3]
    src_img = preprocess(src_img).to(device)

    out_img = equi2equi(src=src_img, rots=rot)
    out_img = postprocess(out_img)
    out_img = np.array(out_img)
    out_path = f"{results_dir}/warping_equi2equi.jpg"
    imwrite(out_path, out_img)

    print(f"Save in {out_path}")


if __name__ == "__main__":
    main()
