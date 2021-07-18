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

### IMPORT NECESSARY DATA FILES ###
#region of interest
studyRegion = '/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/GeoJSONS/drc_training_new.geojson'
studyRegion = gpd.read_file(studyRegion)
print(studyRegion.head())
print(studyRegion.crs)

#landcover data
ESA_CCI = '/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/ESA_CCI/TIF/ESA_CCI_LC_Map_2013.tif'
landcover = xr.open_rasterio(ESA_CCI)
print(landcover)

#landcover classes
landcover_classes = '/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/ESA_CCI/TIF/ESACCI-LC-Legend.csv'
classes = pd.read_csv(landcover_classes, delimiter=";", index_col=0)

print(f"There are {len(classes)} classes.")
print(classes.head())

###  REPROJECT AND CLIP THE LANDCOVER DATA TO STUDY REGION COORDINATE PROJECTION AND BOUNDARIES ###
drc_landcover = landcover.rio.clip(studyRegion.geometry.apply(mapping))
#define projection for DRC
crs_drc = CRS.from_string('EPSG:3341')
landcover_drc_crs = drc_landcover.rio.reproject(crs_drc)
#check the conversion has run okay
landcover_drc_crs.rio.crs
landcover_drc_crs.shape
#save new array as raster ahead of label reclassification
landcover_drc_crs.rio.to_raster('/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/ESA_CCI_Unmerged.tif')
