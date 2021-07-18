import rasterio
from rasterio.plot import show
from rasterio.merge import merge
from rasterio.plot import show
import rasterio.features
import rasterio.warp
import glob
import os
import rioxarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
from rasterio.enums import Resampling
from rasterio.crs import CRS
import matplotlib.pyplot as plt
import earthpy as et
import earthpy.plot as ep
from shapely.geometry import mapping
import subprocess
from osgeo import gdal
import multiprocessing as mp
from typing import List, Any, Sequence, Tuple
import xarray as xarray
from numpy import savetxt




#your images may need merging before you can work with them in python. This is because GEE downloads large images in many tiles.
#if this is the case, use: gdal_merge.py -v -of GTiff "/the/files/you/want/merged.tif -o "/the/output/path/to/merged/files.tif

#you can also reproject in the same way as in "prepare_landcover_data" but for me this wouldn't work with the satellite data so I again
#used GDAl in the command line: gdalwarp -of GTiff -s_srs EPSG:4326 -t_srs EPSG:3341 /the/file/you/want/to/reproject.tif /path/to/the/reprojected/file.tif

#import merged and reprojected satellite image
l8_reprojected = '/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/training_tiles/reprojected_l8_train.tif'
l8_data = xr.open_rasterio(l8_reprojected)
### turning all nans into 0 values
l8_data = l8_data.fillna(0)
print(l8_data)

### RESAMPLE SATELLITE DATA ###
#this is in order to match the lower resolution of the landcover data which is 300m

def write_image(arr: np.array, save_path: os.PathLike, **raster_meta) -> None:

    with rasterio.open(save_path, "w", **raster_meta) as target:
        target.write(arr)


def downsample_image(
    image: rasterio.DatasetReader,
    bands: List[int],
    downsample_factor: int = 10.311,
    resampling: Resampling = Resampling.bilinear,
) -> Tuple[np.ndarray, Any]:


    downsampled_image = image.read(
        bands,
        out_shape=(
            int(image.height / downsample_factor),
            int(image.width / downsample_factor),
        ),
        resampling=resampling,
    )

    transform = image.transform * image.transform.scale(
        (image.width / downsampled_image.shape[-1]),
        (image.height / downsampled_image.shape[-2]),
    )

    return downsampled_image, transform


def generate_downsample(
    file_path: os.PathLike,
    downsample_factor: int = 10.311,
    resampling: Resampling = Resampling.bilinear,
    overwrite: bool = False,
) -> None:


    save_path = '/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/L8_training_data/resampled_satellite_nans.tif'

    with rasterio.open(file_path) as image:
        downsampled_image, transform = downsample_image(
            image,
            image.indexes,
            downsample_factor=downsample_factor,
            resampling=resampling,
        )
        nbands, height, width = downsampled_image.shape

    write_image(
        downsampled_image,
        save_path,
        driver="GTiff",
        height=height,
        width=width,
        count=nbands,
        dtype="float64",
        crs=image.crs,
        transform=transform,
        nodata=image.nodata,
    )


