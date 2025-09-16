"""Various plotting utilities."""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


# there is currently a small issue with the Spilhaus() projection
# See, https://github.com/SciTools/cartopy/issues/2542
def get_spilhaus_mask(lon2d, lat2d):
    # Transform coordinates to Spilhaus projection
    x = ccrs.Spilhaus().transform_points(ccrs.PlateCarree(), lon2d, lat2d)

    y = x[..., 1]
    x = x[..., 0]

    # Get projection bounds
    bound = np.mean(abs(np.array(ccrs.Spilhaus().bounds)))

    # Increase boundary threshold and add distance-based masking
    near_bound = np.logical_or(abs(x) > 0.99 * bound, abs(y) > 0.99 * bound)

    # Calculate distances to edges
    edge_dist = np.minimum(
        np.minimum(abs(x - bound), abs(x + bound)),
        np.minimum(abs(y - bound), abs(y + bound)),
    )
    edge_mask = edge_dist < bound * 0.001

    # Original sign change detection
    sign_change = np.zeros_like(x)
    change_sign0 = np.logical_or(
        x[1:, 1:] * x[:-1, :-1] < 0, y[1:, 1:] * y[:-1, :-1] < 0
    )
    change_sign1 = np.logical_or(
        x[1:, :-1] * x[:-1, 1:] < 0, y[1:, :-1] * y[:-1, 1:] < 0
    )
    change_sign = np.logical_or(change_sign0, change_sign1)

    # Expand sign change mask
    sign_change[1:, 1:] = change_sign
    sign_change[:-1, :-1] = np.logical_or(sign_change[:-1, :-1], change_sign)
    sign_change[1:, :-1] = np.logical_or(sign_change[1:, :-1], change_sign)
    sign_change[:-1, 1:] = np.logical_or(sign_change[:-1, 1:], change_sign)

    # Combine masks with additional edge detection
    final_mask = np.logical_or(np.logical_and(near_bound, sign_change), edge_mask)

    # Add additional buffer around masked regions
    final_mask = np.maximum.reduce(
        [
            final_mask,
            np.roll(final_mask, 1, axis=0),
            np.roll(final_mask, -1, axis=0),
            np.roll(final_mask, 1, axis=1),
            np.roll(final_mask, -1, axis=1),
        ]
    )

    return final_mask


def add_colorbar(fig, ax, var, fmt=None, range_limit=None):
    """Colorbar position and format properly"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.03, axes_class=plt.Axes)
    cb = fig.colorbar(var, cax=cax, format=fmt)
    if range_limit:
        cb.mappable.set_clim(range_limit)
    cb.ax.tick_params(which="major", labelsize=6, length=3, width=0.5, pad=0.05)
    return cb
