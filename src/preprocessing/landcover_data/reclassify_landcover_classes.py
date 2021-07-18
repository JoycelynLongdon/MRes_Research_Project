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



#import clipped and reprojected landcover data
training_landcover_path = '/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/landcover_clipped_reproj.tif'
training_landcover = xr.open_rasterio(training_landcover_path)

#define reclassification function
def reclassify_landcover(input_array):
    '''input xarray.DataArray with classes and output dataarray with super classes
        but same coords and attributes as input'''

    new_class_1 = np.where((input_array == 10)|(input_array == 20)|(input_array ==130),
                             1, input_array) #cropland
    new_class_2 = np.where((input_array == 11)|(input_array == 12)|(input_array == 30)|(input_array == 153)|(input_array == 152)|(input_array == 151)|(input_array == 150)|(input_array == 110)|(input_array == 120)|(input_array == 121)|(input_array == 122),
                             2, new_class_1)#shrubland
    #new_class_3 = np.where((input_array == 40),
                             #3, new_class_2)#mosaic vegetation
    new_class_3 = np.where((input_array == 50)|(input_array == 60)|(input_array == 40)|(input_array == 61)|(input_array == 62)|(input_array == 100)|(input_array == 151)|(input_array == 70)|(input_array == 71)|(input_array == 72)|(input_array == 80)|(input_array == 81)|(input_array == 82)|(input_array == 90)|(input_array == 160)|(input_array == 170),
                             3, new_class_2)#forest
    #new_class_3 = np.where((input_array == 11)|(input_array == 12)|(input_array == 40)|,
                             #3, new_class_2)#shrubland
    #new_class_4 = np.where((input_array ==130),
                             #4, new_class_3) #grassland
    new_class_4 = np.where((input_array == 190)|(input_array == 202)|(input_array == 201)|(input_array == 200),
                             4, new_class_3) #urban
    new_class_5 = np.where((input_array == 210)|(input_array == 180),
                             5, new_class_4)#water
    new_class_6 = np.where((input_array == 0)|(input_array == 140)|(input_array == 220),
                             6, new_class_5)#No data and Other
    #new_class_10 = np.where((input_array == 0), 10, new_class_9) #no data


    output_array_final = xarray.DataArray(data=new_class_6, coords=input_array.coords, attrs=input_array.attrs)

    return output_array_final

#new classes
drc_training_landcover = reclassify_landcover(training_landcover)
superclass_data=[]
superclass_vals = np.unique(drc_training_landcover)
for c in superclass_vals:
    superclass_data.append((drc_training_landcover == c).sum())

superclass_df = pd.DataFrame(superclass_data, index = superclass_vals, columns = ['pixel_count'])


superclass_df.insert(superclass_df.shape[1], 'class_name', ['Cropland', 'Shrubland', 'Forest',
                         'Urban', 'Water', 'No Data and Other'
                                                           ])

superclass_df

#save reclassified data as a new raster ready for RF classification
drc_training_landcover.rio.to_raster('/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/classification_training_data/third_remerge_landcover_training_data.tif')
