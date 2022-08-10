import numpy as np
from typing import List, Tuple

def compute_roi_from_ply_list(clouds_list: List[str], resolution: float):
    """Projects a points cloud into the raster band(s) of a raster image (points clouds as files)

    Positional arguments:
        clouds_list -- list of cloud.ply files
        resolution -- resolution of the georeferenced output raster file
        roi -- region of interest: (xoff, yoff, xsize, ysize), compute plyextrema if None

    """
    xmin, xmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf

    for cloud in clouds_list:
        cloud_data, _ = utils.read_3d_point_cloud_from_ply(cloud)
        current_cloud = cloud_data.astype(np.float64)
        xx, yy = current_cloud[:, 0], current_cloud[:, 1]
        xmin = np.min((xmin, np.amin(xx)))
        ymin = np.min((ymin, np.amin(yy)))
        xmax = np.max((xmax, np.amax(xx)))
        ymax = np.max((ymax, np.amax(yy)))

    xoff = np.floor(xmin / resolution) * resolution
    xsize = int(1 + np.floor((xmax - xoff) / resolution))

    yoff = np.ceil(ymax / resolution) * resolution
    ysize = int(1 - np.floor((ymin - yoff) / resolution))

    return xoff, yoff, xsize, ysize

def plyflatten_from_plyfiles_list(clouds_list: List[str], resolution: float, radius=0, roi: Tuple[float, float, float, float]=None, sigma=None, std=False, min=False, max=False):
    """Projects a points cloud into the raster band(s) of a raster image (points clouds as files)

    Parameters
    ----------
        clouds_list -- list of cloud.ply files
        resolution -- resolution of the georeferenced output raster file
        roi -- region of interest: (xoff, yoff, xsize, ysize), compute plyextrema if None

    Returns
    -------
    """
    # NOTE: region of interest (compute plyextrema if roi is None)
    xoff, yoff, xsize, ysize = compute_roi_from_ply_list(clouds_list, resolution) if roi is None else roi