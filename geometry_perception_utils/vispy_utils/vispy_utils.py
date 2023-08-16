import numpy as np
import vispy
from functools import partial
from vispy.scene import visuals
from vispy.visuals.transforms import STTransform
from vispy.scene.visuals import Text
from vispy import app, scene, io
import sys
from matplotlib.colors import hsv_to_rgb
import vispy.io as vispy_file
import os
from imageio import imwrite



def plot_list_pcl(list_pcl, size=1, scale_factor=15, fn=None, return_png=False):

    colors = get_color_list(number_of_colors=list_pcl.__len__())
    pcl_colors = []
    for pcl, c in zip(list_pcl, colors.T):
        pcl_colors.append(np.ones_like(pcl)*c.reshape(3, 1))

    return plot_color_plc(np.hstack(list_pcl).T, color=np.hstack(pcl_colors).T, size=size, scale_factor=scale_factor, fn=fn, return_png=return_png)


def get_color_list(array_colors=None, fr=0.1, return_list=False, number_of_colors=None):
    """
    Returns a different color RGB for every element in the array_color
    """
    if array_colors is not None:
        number_of_colors = len(array_colors)

    h = np.linspace(0.1, 0.8, number_of_colors)
    # np.random.shuffle(h)
    # values = np.linspace(0, np.pi, number_of_colors)
    colors = np.ones((3, number_of_colors))

    colors[0, :] = h

    return hsv_to_rgb(colors.T).T


def setting_viewer(main_axis=True, bgcolor="black", caption=""):
    canvas = vispy.scene.SceneCanvas(keys="interactive", show=True, bgcolor=bgcolor)
    size_win = 1024
    canvas.size = 2 * size_win, size_win

    t1 = Text(caption, parent=canvas.scene, color="white")
    t1.font_size = 24
    t1.pos = canvas.size[0] // 2, canvas.size[1] // 10

    view = canvas.central_widget.add_view()
    view.camera = "arcball"  # turntable / arcball / fly / perspective

    if main_axis:
        visuals.XYZAxis(parent=view.scene)

    return view, canvas


def setting_pcl(view, size=5, edge_width=2, antialias=0):
    scatter = visuals.Markers()
    scatter.set_gl_state(
        "translucent",
        depth_test=True,
        blend=True,
        blend_func=("src_alpha", "one_minus_src_alpha"),
    )
    # scatter.set_gl_state(depth_test=True)
    scatter.antialias = 0
    view.add(scatter)
    return partial(scatter.set_data, size=size, edge_width=edge_width)


def plot_color_plc(
    points,
    color=(1, 1, 1, 1),
    size=0.5,
    plot_main_axis=True,
    background="black",
    scale_factor=100,
    caption="",
    fn=None,
    return_png=False,
):

    view, canvas= setting_viewer(main_axis=plot_main_axis, bgcolor=background, caption=caption)
    view.camera = vispy.scene.TurntableCamera(
        elevation=90, azimuth=90, roll=0, fov=0, up="-y"
    )
    # view.camera = vispy.scene.TurntableCamera(elevation=90,
    #                                           azimuth=0,
    #                                           roll=0,
    #                                           fov=0,
    #                                           up='-y')
    view.camera.scale_factor = scale_factor
    draw_pcl = setting_pcl(view=view)
    draw_pcl(points, edge_color=color, size=size)
    if return_png:
        return canvas.render()
    if fn is not None:
        imwrite(fn, canvas.render())
    vispy.app.run()
    # return canvas.render()
