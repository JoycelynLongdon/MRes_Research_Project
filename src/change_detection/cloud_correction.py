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
import seaborn as sns

## code to correct for any cloud pixels

#test
image_1 = np.load('/gws/nopw/j04/ai4er/users/jl2182/data/Figures/PIREDD_Classified/Corrected_Classified/PIREDD_Plataue_L8_2014_corrected_classified.npy')
image_2 = np.load('/gws/nopw/j04/ai4er/users/jl2182/data/Figures/PIREDD_Classified/PIREDD_Plataue_L8_2015_classified.npy')
image_3 = np.load('/gws/nopw/j04/ai4er/users/jl2182/data/Figures/PIREDD_Classified/PIREDD_Plataue_L8_2016_classified.npy')

image_1 = image_1.astype(int)
image_2 = image_2.astype(int)
image_3 = image_3.astype(int)

#visualise landcover maps
# See https://github.com/matplotlib/matplotlib/issues/844/
n = image_1.max()
# Next setup a colormap for our map
colors = dict((
    (1, (111, 97, 6,255)),  # Cropland (brown)
    (2, (135, 198, 42,255)),  # Shrubland (light green)
    (3, (15, 91, 3,255)),  # Forest (dark green)
    (4, (255, 26, 0,255)),  # Urban (red)
    (5, (0, 0, 255,255)),  # Water (blue)
    (6, (0, 0, 0,0)) #No Data/Clouds
))
# Put 0 - 255 as float 0 - 1
for k in colors:
    v = colors[k]
    _v = [_v / 255.0 for _v in v]
    colors[k] = _v

index_colors = [colors[key] if key in colors else
                (255, 255, 255, 0) for key in range(1, n + 1)]
cmap = plt.matplotlib.colors.ListedColormap(index_colors, 'Classification', n)

# Now show the classmap next to the image
plt.figure(figsize = (8,8))

plt.subplot(131)
plt.title('Year 1')
plt.imshow(image_1, cmap=cmap, interpolation='none')

plt.subplot(132)
plt.title('Year 2')
plt.imshow(image_2, cmap=cmap, interpolation='none')

plt.subplot(133)
plt.title('Year 3')
plt.imshow(image_2, cmap=cmap, interpolation='none')

plt.show()

#account for years where pixels might be covered by cloud
A = np.ones(image_1.shape)
for i in range(image_2.shape[0]):
    for j in range(image_2.shape[1]):
        #forest, cloud, forest
        if image_1[i,j] == 3 and image_2[i,j] == 6 and image_3[i,j] == 3:
            A[i,j] = image_1[i,j] #no change
        #non-forest, cloud, non-forest
        if image_1[i,j] !=3 and image_2[i,j] == 6 and image_3[i,j] != 3:
            A[i,j] = image_1[i,j] #change
        #forest, cloud, not forest - saying change happens in the year that the pixel is clouded over
        if image_1[i,j] == 3 and image_2[i,j] == 6 and image_3[i,j] != 3:
            A[i,j] = image_3[i,j]
        #clear pixel, non-clear pixel
        if image_1[i,j] != 6 and image_2[i,j] == 6:
            A[i,j] = image_1[i,j]
        else:
            A[i,j] = image_2[i,j] #blank out all other pixels/changes

#a similar method is taken in Landtrendr where a decrease in the observed spectral value is ignored if the
#following year it returns to the level of the previous year, which would indicate cloud cover
#limitation here is that if the image 1 and image 3 pixel classifcation are different, then bias has been created about which
#class the pixel will become which could skew which year the change event was seen to happen

#compare before and after
n = image_1.max()
# Next setup a colormap for our map
colors = dict((
    (1, (111, 97, 6,255)),  # Cropland (brown)
    (2, (135, 198, 42,255)),  # Shrubland (light green)
    (3, (15, 91, 3,255)),  # Forest (dark green)
    (4, (255, 26, 0,255)),  # Urban (red)
    (5, (0, 0, 255,255)),  # Water (blue)
    (6, (0, 0, 0,0)) #No Data/Clouds
))
# Put 0 - 255 as float 0 - 1
for k in colors:
    v = colors[k]
    _v = [_v / 255.0 for _v in v]
    colors[k] = _v

index_colors = [colors[key] if key in colors else
                (255, 255, 255, 0) for key in range(1, n + 1)]
cmap = plt.matplotlib.colors.ListedColormap(index_colors, 'Classification', n)


plt.figure(figsize = (20,25))

plt.subplot(121)
plt.title('Before Correction')
plt.imshow(image_2, cmap=cmap, interpolation='none')

#plt.figure(figsize = (8,8))
plt.subplot(122)
plt.title('After Correction')
plt.imshow(A, cmap=cmap, interpolation='none')
