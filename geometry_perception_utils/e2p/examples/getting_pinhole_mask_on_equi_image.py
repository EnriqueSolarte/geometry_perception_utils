
from geometry_perception_utils.e2p import pinhole_mask_on_equirectangular
from  imageio import imwrite
from pathlib import Path  

if __name__ == "__main__":
    """
    fov = 90 # FOV of the perspective camera
    u_deg=0 # yaw angle of the perspective camera 
    v_deg=0 # pitch angle of the perspective camera
    out_hw = (512, 1024) # output equirectangular image size
    """
    
    img = pinhole_mask_on_equirectangular(u_deg=0, v_deg=0, out_hw=(512, 1024)) 
    fn = f"{Path(__file__).parent.__str__()}/pinhole_mask.jpg"
    imwrite(f'{fn}', img)
    print(img.shape)
    
    

    

    
    