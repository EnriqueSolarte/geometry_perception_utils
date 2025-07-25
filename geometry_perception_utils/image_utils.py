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
from geometry_perception_utils.vispy_utils import get_color_list as vispy_get_color_list


def get_color_list(array_colors=None, fr=0.1, return_list=False, number_of_colors=None):
    return vispy_get_color_list(array_colors, fr, return_list, number_of_colors)


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


def get_default_uv_map(shape):
    h, w = shape
    u = np.linspace(0, w-1, w).astype(np.int16)
    v = np.linspace(0, h-1, h).astype(np.int16)
    uu, vv = np.meshgrid(u, v)
    uv_map = np.stack((vv, uu)).transpose(1, 2, 0)
    return uv_map


def set_alpha(img, alpha=0.5):
    img = img * alpha + (1 - alpha) * 255
    return img.astype(np.uint8)


def get_map_from_array(array, shape):
    """
    returns a map (H, W, 3) from an array (3, n), or a map (H, W) from (1, n). 
    This function is the inverse of get_color_array(*)
    """
    H, W = shape[:2]
    channels = [a.reshape((H, W)) for a in array]
    return np.stack(channels, axis=-1)


def get_color_array(color_map):
    """
    returns an array (3, n) of map (H, W, 3), or (1, n) of map (H, W)
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
    H, W = image.shape[:2]
    if image.shape.__len__() == 3:
        for i in np.linspace(-size//2, size//2, size*2):
            for j in np.linspace(-size//2, size//2, size*2):
                # image[(boundary_uv[1] + i) % image.shape[0],
                #       boundary_uv[0], :] = np.array(color)
                image[np.int16(boundary_uv[1]+i) %
                      H, np.int16((boundary_uv[0]+j) % W), :] = color

            # image[(boundary_uv[1]-i)% 0, boundary_uv[0], :] = np.array(color)
    else:
        for i in np.linspace(-size//2, size//2, size*2):
            for j in np.linspace(-size//2, size//2, size*2):
                image[np.int16(boundary_uv[1]+i) %
                      H, np.int16((boundary_uv[0]+j) % W)] = np.max(color)

            # image[(boundary_uv[1] + i) % image.shape[0], boundary_uv[0]] = 255
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

    # image = draw_sorted_boundaries_uv(image=image,
    #                    boundary_uv=uv_ceiling,
    #                    color=color,
    #                    size=size)
    # image = draw_sorted_boundaries_uv(image=image,
    #                    boundary_uv=uv_floor,
    #                    color=color,
    #                    size=size)
    return image


def draw_boundaries_xyz(image, xyz, color=(0, 255, 0), size=2):
    uv = xyz2uv(xyz)
    return draw_boundaries_uv(image, uv, color, size)


def add_caption_to_image(image, caption, position=(20, 20), color=(255, 0, 0), font_s=100):
    img_obj = Image.fromarray(image.astype(np.uint8))
    img_draw = ImageDraw.Draw(img_obj)
    font_obj = ImageFont.truetype("FreeMono.ttf", font_s)
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


def draw_att_map_on_image(image, att_map):
    assert att_map.shape[:2] == image.shape[:2]
    mask = att_map/att_map.max()
    vis = (mask * 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    vis = image.copy() * 0.2 + vis * 0.6
    vis = vis.astype(np.uint8)[:, :, :: -1]
    return vis


def draw_mask_on_image(image, mask):
    assert mask.shape[:2] == image.shape[:2]
    vis = image.copy() * 0.4
    vis[mask > 0] = vis[mask > 0] // 2 + \
        np.array([10, 255, 10], dtype=np.uint8) // 2
    return vis.astype(np.uint8)


def uniform_sampling(h, w, stride=5):
    u = np.linspace(0, w - 1, w//stride).astype(int)
    v = np.linspace(0, h - 1, h//stride).astype(int)
    uu, vv = np.meshgrid(u, v)
    mask = np.zeros((h, w), dtype=bool)
    mask[vv.flatten(), uu.flatten()] = True
    return mask
