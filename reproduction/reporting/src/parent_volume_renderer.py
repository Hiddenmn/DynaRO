"""Full-parent rendering helpers, copied without numerical or visual edits."""
from __future__ import annotations
import gc
import itertools
import numpy as np
from surface_renderer import (make_camera, project, mesh, geometry_buffer,
                              light_pixels, PALETTE, LIGHT)

PIXELS = 1050
CAMERA = {"azimuth": -125., "elevation": 24., "projection": "orthographic"}
INK = "#253746"

def wire(ax,c,depth):
    corners=np.asarray(list(itertools.product((-1.,128.),repeat=3)))
    for i in range(8):
        for j in range(i):
            if np.count_nonzero(corners[i]!=corners[j])!=1:
                continue
            points=corners[i]+np.linspace(0,1,300)[:,None]*(corners[j]-corners[i])
            q=project(points,c)
            ix=np.clip(q[:,0].astype(int),0,c["width"]-1)
            iy=np.clip(q[:,1].astype(int),0,c["height"]-1)
            visible=q[:,2]>=depth[iy,ix]-.25
            xy=q[:,:2].copy()
            xy[~visible]=np.nan
            ax.plot(xy[:,0],xy[:,1],lw=.45,color="#c8d1d6")


def render(mask, *, water=None, mobilized=False, ghost=None):
    camera = make_camera(PIXELS, az=CAMERA["azimuth"], el=CAMERA["elevation"])
    m = mesh(mask, 6 if mobilized else 7)
    depth, normals, categories = geometry_buffer(m, camera)
    light_camera = make_camera(700, direction=LIGHT)
    shadow, _, _ = geometry_buffer(m, light_camera)
    rgb, _ = light_pixels(
        depth, normals, categories, mask, PALETTE,
        camera["center"], camera["right"], camera["up"], camera["view"], camera["span"],
        LIGHT, shadow, light_camera["center"], light_camera["right"], light_camera["up"],
        light_camera["view"], light_camera["span"])
    oil_rgb = rgb.copy()
    if water is not None and np.any(water):
        water_mesh = mesh(water, 2)
        water_depth, water_normals, water_category = geometry_buffer(water_mesh, camera)
        visible = (water_category == 2) & (water_depth >= depth)
        shading = .74 + .26*np.maximum(water_normals@LIGHT, 0)
        water_rgb = PALETTE[2][None, None, :]*shading[..., None]
        rgb[visible] = .80*rgb[visible] + .20*water_rgb[visible]
    if ghost is not None:
        grey = np.asarray([.2126, .7152, .0722], np.float32)
        ghost_grey = ghost@grey
        faint_initial = 1 - .17*(1-ghost_grey)
        background = categories == 0
        rgb[background] = faint_initial[background, None]
    result = {"rgb": np.clip(rgb, 0, 1), "oil_rgb": oil_rgb, "camera": camera, "depth": depth,
              "surface_voxels": int(mask.sum()), "triangles": len(m["faces"])}
    del m, normals, categories, shadow
    gc.collect()
    return result


def volume_panel(fig, bounds, rendered, panel_id, title, metric=None):
    ax = fig.add_axes(bounds, label=panel_id)
    ax.imshow(rendered["rgb"], interpolation="none")
    wire(ax, rendered["camera"], rendered["depth"])
    ax.axis("off")
    ax.text(.015, 1.035, panel_id, transform=ax.transAxes, weight="bold", fontsize=9,
            ha="left", va="bottom", color="#253746")
    ax.text(.11, 1.035, title, transform=ax.transAxes, fontsize=8,
            ha="left", va="bottom", color="#253746")
    if metric:
        ax.text(.5, -.025, metric, transform=ax.transAxes, fontsize=7.3,
                ha="center", va="top", color="#526474")
    return ax


def header_frame(fig, bounds, panel_id, title):
    ax = fig.add_axes(bounds, label=panel_id)
    ax.axis("off")
    ax.text(.015, 1.035, panel_id, transform=ax.transAxes, weight="bold",
            fontsize=9, ha="left", va="bottom", color=INK)
    ax.text(.11, 1.035, title, transform=ax.transAxes, fontsize=8,
            ha="left", va="bottom", color=INK)
    return ax


def plot_style(ax):
    ax.tick_params(labelsize=7, length=2.5, pad=2)
    ax.xaxis.label.set_size(7.3)
    ax.yaxis.label.set_size(7.3)
    ax.grid(axis="y", color="#e4e9ee", linewidth=.5)

