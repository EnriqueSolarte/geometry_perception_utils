from imageio import imread
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import skimage.filters
import cv2
from geometry_perception_utils.spherical_utils import phi_coords2xyz, xyz2uv
from skimage.transform import rescale, resize, downscale_local_mean

class colors:    
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    MAGENTA = (255, 0, 255)
    CYAN = (0, 255, 255)
    BLUE = (51, 51, 255)
    YELLOW = (255, 255, 0)
    ORANGE = (255, 128, 0)
    PINK = (255, 0, 128)
    PURPLE = (128, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)


COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_BLUE = (51, 51, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_ORANGE = (255, 128, 0)
COLOR_PINK = (255, 0, 128)
COLOR_PURPLE = (128, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


def set_alpha(img, alpha=0.5):
    img = img * alpha + (1 - alpha) * 255
    return img.astype(np.uint8)
   

def get_color_array(color_map):
    """
    returns an array (3, n) of the colors in image (H, W)
    """
    # ! This is the same solution by flatten every channel
    if len(color_map.shape) > 2:
        return np.vstack((
            color_map[:, :, 0].flatten(),
            color_map[:, :, 1].flatten(),
            color_map[:, :, 2].flatten(),
        ))
    else:
        return np.vstack(
            (color_map.flatten(), color_map.flatten(), color_map.flatten()))


def load_depth_map(fpath):
    """Make sure the depth map has shape (H, W) but not (H, W, 1)."""
    depth_map = np.array(imread(fpath))
    if depth_map.shape[-1] == 1:
        depth_map = depth_map.squeeze(-1)
    return depth_map


def draw_sorted_boundaries_uv(image, boundary_uv, color=(0, 255, 0), size=2):
    boundary_uv = boundary_uv[:, np.argsort(boundary_uv[0, :])]
    [
        cv2.line(image, (boundary_uv[0, i], boundary_uv[1, i]),
                 (boundary_uv[0, i + 1], boundary_uv[1, i + 1]), color, size)
        for i in range(0, boundary_uv.shape[1] - 1)
    ]
    return image


def draw_boundaries_uv(image, boundary_uv, color=(0, 255, 0), size=2):
    if image.shape.__len__() == 3:
        for i in range(size):
            image[(boundary_uv[1] + i) % image.shape[0],
                  boundary_uv[0], :] = np.array(color)
            # image[(boundary_uv[1]-i)% 0, boundary_uv[0], :] = np.array(color)
    else:
        for i in range(size):
            image[(boundary_uv[1] + i) % image.shape[0], boundary_uv[0]] = 255
            # image[(boundary_uv[1]-i)% 0, boundary_uv[0]] = 255

    return image


def draw_boundaries_phi_coords(image, phi_coords, color=(0, 255, 0), size=2):
    # ! Compute bearings
    theta_coords = np.linspace(-np.pi, np.pi, phi_coords.shape[1])
    bearings_ceiling = phi_coords2xyz(
        phi_coords=phi_coords[0, :], theta_coords=theta_coords)
    bearings_floor = phi_coords2xyz(
        phi_coords=phi_coords[1, :], theta_coords=theta_coords)

    uv_ceiling = xyz2uv(bearings_ceiling)
    uv_floor = xyz2uv(bearings_floor)

    draw_boundaries_uv(image=image,
                       boundary_uv=uv_ceiling,
                       color=color,
                       size=size)
    draw_boundaries_uv(image=image,
                       boundary_uv=uv_floor,
                       color=color,
                       size=size)

    return image


def draw_boundaries_xyz(image, xyz, color=(0, 255, 0), size=2):
    uv = xyz2uv(xyz)
    return draw_boundaries_uv(image, uv, color, size)


def add_caption_to_image(image, caption, position=(20, 20), color=(255, 0, 0)):
    img_obj = Image.fromarray(image.astype(np.uint8))
    img_draw = ImageDraw.Draw(img_obj)
    font_obj = ImageFont.truetype("FreeMono.ttf", 20)
    img_draw.text((position[1], position[0]),
                  f"{caption}", font=font_obj, fill=color)
    return np.array(img_obj)


def plot_image(image, caption, figure=0):
    plt.figure(figure)
    plt.clf()
    image = add_caption_to_image(image, caption)
    plt.imshow(image)
    plt.draw()
    plt.waitforbuttonpress(0.01)


def draw_uncertainty_map(sigma_boundary, peak_boundary, shape=(512, 1024)):
    H, W = shape
    img_map = np.zeros((H, W))
    for u, v, sigma in zip(peak_boundary[0, :], peak_boundary[1, :],
                           sigma_boundary):
        sigma_bon = (sigma / np.pi) * shape[0]

        sampled_points = np.random.normal(v, sigma_bon, 512).astype(np.int16)
        sampled_points[sampled_points >= H] = H - 1
        sampled_points[sampled_points <= 0] = 0

        u_point = (np.ones_like(sampled_points) * u).astype(np.int16)
        img_map[sampled_points, u_point] = 1

    img_map = skimage.filters.gaussian(img_map,
                                       sigma=(15, 5),
                                       truncate=5,
                                       channel_axis=True)

    img_map = img_map / img_map.max()
    return img_map


def hmerge_list_images(list_images):
    min_h = np.min([img.shape[0] for img in list_images])
    scales = [min_h/img.shape[0] for img in list_images]
    __images = []
    for img, sc in zip(list_images, scales):
        resize_img = rescale(
            img, scale=sc, anti_aliasing=True, channel_axis=True)
        __images.append(resize_img)
    return np.hstack(__images)


def vmerge_list_images(list_images):
    min_w = np.min([img.shape[1] for img in list_images])
    scales = [min_w/img.shape[0] for img in list_images]
    __images = []
    for img, sc in zip(list_images, scales):
        resize_img = rescale(
            img, scale=sc, anti_aliasing=True, channel_axis=True)
        __images.append(resize_img)
    return np.vstack(__images)
