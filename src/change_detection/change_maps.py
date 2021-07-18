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
from matplotlib.colors import colorConverter
import scipy
import scipy.ndimage




##code to detect change in landcover classified images and compare with established methods

#load landcover maps
image_1 = np.load('/gws/nopw/j04/ai4er/users/jl2182/data/Figures/PIREDD_Classified/Corrected_Classified/PIREDD_Plataue_L8_2013_classified.npy')
image_2 = np.load('/gws/nopw/j04/ai4er/users/jl2182/data/Figures/PIREDD_Classified/Corrected_Classified/PIREDD_Plataue_L8_2015_corrected_classified.npy')

image_1 = image_1.astype(int)
image_2 = image_2.astype(int)

#load hansen GFC data
Hansen = np.load('/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/Hansen_Results/loss_year/resampled/loss_year_19.npy')

#load LandTrendr data and pull up change points for a specific year and resample
LandTrendr_path = '/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/LandTrendr_Results/PIREDD_Plateau/Change_Maps/PIREDD_Plateau_year_of_change_map.tif'
LandTrendr = xr.open_rasterio(LandTrendr_path)
LandTrendr_2014 = np.where(LandTrendr == 2014,1,0)
LandTrendr_2014_Resampled = scipy.ndimage.zoom(LandTrendr_2014[0],0.0974786, order=0)
LandTrendr_2014_Resampled.shape

#forest to shrubland change
f_s = np.ones(image_1.shape)
for i in range(image_1.shape[0]):
    for j in range(image_1.shape[1]):
        if image_1[i,j] == 3 and image_1[i,j] == image_2[i,j]:
            f_s[i,j] = 0 #no change
        if image_1[i,j] == 3 and image_2[i,j] !=3:
            f_s[i,j] = 1 #change
        else:
            f_s[i,j] = 0 #blank out all other pixels/changes

#forest to cropland change
f_c = np.ones(image_1.shape)
for i in range(image_1.shape[0]):
    for j in range(image_1.shape[1]):
        if image_1[i,j] == 3 and image_1[i,j] == image_2[i,j]:
            f_c[i,j] = 0 #no change
        if image_1[i,j] == 3 and image_2[i,j] ==1:
            f_c[i,j] = 1 #change
        else:
            f_c[i,j] = 0 #blank out all other pixels/changes

#shrubland to cropland change
s_c = np.ones(image_1.shape)
for i in range(image_1.shape[0]):
    for j in range(image_1.shape[1]):
        if image_1[i,j] == 2 and image_1[i,j] == image_2[i,j]:
            s_c[i,j] = 0 #no change
        if image_1[i,j] == 2 and image_2[i,j] == 1:
            s_c[i,j] = 1 #change
        else:
            s_c[i,j] = 0 #blank out all other pixels/changes

#plot satellite image as base map
# Open the file:
raster = rasterio.open('/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/PIREDD_Test_Data/Filled/PIREDD_Plataue_L8_2013_filled.tif')

# Normalize bands into 0.0 - 1.0 scale
def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

# Convert to numpy arrays
nir = raster.read(4)
red = raster.read(3)
green = raster.read(2)

# Normalize band DN
nir_norm = normalize(nir)
red_norm = normalize(red)
green_norm = normalize(green)

# Stack bands
nrg = np.dstack((nir_norm, red_norm, green_norm))

#set colour maps
colour_1 = colorConverter.to_rgba('white',alpha=0.0)
colour_2 = 'blue'
colour_3 = colorConverter.to_rgba('white',alpha=0.0)
cmap1 = plt.matplotlib.colors.ListedColormap([colour_1,colour_2, colour_3])

colour_4 = colorConverter.to_rgba('white',alpha=0.0)
colour_5 = 'yellow'
colour_6 = colorConverter.to_rgba('white',alpha=0.0)
cmap2 = plt.matplotlib.colors.ListedColormap([colour_4, colour_5, colour_6])

colour_7 = colorConverter.to_rgba('white',alpha=0.0)
colour_8 = 'red'
colour_9 = colorConverter.to_rgba('white',alpha=0.0)
cmap3 = plt.matplotlib.colors.ListedColormap([colour_7, colour_8, colour_9])

colour_10 = colorConverter.to_rgba('white',alpha=0.0)
colour_11 = 'orange'
colour_12 = colorConverter.to_rgba('white',alpha=0.0)
cmap4 = plt.matplotlib.colors.ListedColormap([colour_10, colour_11, colour_12])

#plot and save change maps
plt.figure(figsize = (10,10))
plt.imshow(nrg,alpha = 0.75)
plt.imshow(Hansen, cmap = cmap1)
plt.imshow(LandTrendr_2014_Resampled, cmap = cmap4)
plt.imshow(f_c, cmap = cmap2)#orange
plt.imshow(s_c, cmap = cmap2)#yellow
#plt.savefig('/gws/nopw/j04/ai4er/users/jl2182/data/Figures/LC_LT_GFC_Compariosn/2019_comparison.png')
#plt.savefig('/gws/nopw/j04/ai4er/users/jl2182/data/Figures/LC_LT_GFC_Compariosn/2019_comparison.tif')
