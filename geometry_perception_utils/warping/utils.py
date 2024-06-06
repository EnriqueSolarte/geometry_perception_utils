
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import Union
import cv2


def preprocess(
    img: Union[np.ndarray, Image.Image], is_cv2: bool = False
) -> torch.Tensor:
    """Preprocesses image"""
    if isinstance(img, np.ndarray) and is_cv2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if isinstance(img, Image.Image):
        # Sometimes images are RGBA
        img = img.convert("RGB")

    to_tensor = transforms.Compose([transforms.ToTensor()])
    img = to_tensor(img)
    assert len(img.shape) == 3, "input must be dim=3"
    assert img.shape[0] == 3, "input must be HWC"
    return img


def postprocess(
    img: torch.Tensor, to_cv2: bool = False
) -> Union[np.ndarray, Image.Image]:
    if to_cv2:
        img = np.asarray(img.to("cpu").numpy() * 255, dtype=np.uint8)
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    else:
        to_PIL = transforms.Compose([transforms.ToPILImage()])
        img = img.to("cpu")
        img = to_PIL(img)
        return img
