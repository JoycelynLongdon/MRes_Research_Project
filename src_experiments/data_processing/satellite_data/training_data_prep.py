#!/usr/bin/env python
# coding: utf-8

# In[1]:


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




get_ipython().run_line_magic('matplotlib', 'inline')


# # Landcover Data

# ### Importing the Training Region Polygon

# In[2]:


#region of interest
studyRegion = '/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/GeoJSONS/drc_training_new.geojson'
studyRegion = gpd.read_file(studyRegion)
print(studyRegion.head())
print(studyRegion.crs)


# In[ ]:


fig, ax = plt.subplots(figsize=(6, 6))

studyRegion.plot(ax=ax)

ax.set_title("studyRegion",
             fontsize=16)
plt.show()


# ### Importing and Exploring the Landcover Data

# In[ ]:


#landcover data
ESA_CCI = '/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/ESA_CCI/TIF/ESA_CCI_LC_Map_2013.tif'
landcover = xr.open_rasterio(ESA_CCI)
print(landcover)


# In[ ]:


#landcover classes
landcover_classes = '/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/ESA_CCI/TIF/ESACCI-LC-Legend.csv'
classes = pd.read_csv(landcover_classes, delimiter=";", index_col=0)

print(f"There are {len(classes)} classes.")
print(classes.head())


# In[ ]:


#explore statistics
print(landcover.rio.crs)
print(landcover.rio.nodata)
print(landcover.rio.bounds())
print(landcover.rio.width)
print(landcover.rio.height)
print(landcover.rio.crs.wkt)


# In[ ]:


#landcover.values
landcover


# ### Reproject and Clip the Landcover Data to DRC Cooridnate Projection and Study Region

# In[ ]:


drc_landcover = landcover.rio.clip(studyRegion.geometry.apply(mapping))


# In[ ]:


#define projection for DRC
crs_drc = CRS.from_string('EPSG:3341')
landcover_drc_crs = drc_landcover.rio.reproject(crs_drc)


# In[ ]:


landcover_drc_crs.rio.crs
landcover_drc_crs.shape


# In[ ]:


landcover_drc_crs.values


# In[ ]:


#save as raster ahead of reclassification
landcover_drc_crs.rio.to_raster('/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/ESA_CCI_Unmerged.tif')


# ### Reclassify classes

# In[8]:


#training_landcover_path = '/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/ESA_CCI_Unmerged.tif'
#training_landcover = xr.open_rasterio(training_landcover_path)


# In[ ]:


training_landcover.plot()


# In[6]:


training_landcover_path = '/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/landcover_clipped_reproj.tif'
training_landcover = xr.open_rasterio(training_landcover_path)

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

#super classes
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


# In[7]:


drc_training_landcover.rio.to_raster('/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/classification_training_data/third_remerge_landcover_training_data.tif')


# In[ ]:


savetxt('/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/drc_training_landcover', drc_training_landcover, delimiter=',')


# In[ ]:


drc_training_landcover


# In[ ]:


#from 3D ----> 2D
drc_training_landcover = landcover_drc_crs[0, :, :]
print(drc_training_landcover.shape)


# In[ ]:



# save to csv file
savetxt('/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/ESA_Landcover_Data.csv', ESA_Landcover_Data, delimiter=',')


# In[ ]:


landcover_drc_crs.values


# In[ ]:


# ESA_Landcover_Data.values
ESA_Landcover_Data


# In[ ]:


#print(landcover_drc_crs)


# In[ ]:


f, ax = plt.subplots(figsize=(20, 10))
landcover_drc_crs.plot(ax=ax)
ax.set(title="Landcover Map of Training Region")
ax.set_axis_off()
plt.savefig('/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/landcover_map_of_training_region.png')
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(20, 10))
drc_training_landcover.plot(ax=ax)
ax.set(title="Landcover Map of Training Region")
ax.set_axis_off()
plt.savefig('/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/landcover_map_of_training_region.png')
plt.show()


# # Satellite Data

# ### Importing and Exploring the Satellite Data
#

# In[ ]:


#l8 = '/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/training_tiles/merged_l8_train_data.tif'
l8_reprojected = '/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/training_tiles/reprojected_l8_train.tif' #the imagery was reprojected in
#the command line using gdal as rioxarray was re-filloing the array with unwanted values
l8_data = xr.open_rasterio(l8_reprojected)
print(l8_data)


# In[ ]:


#explore statistics
print(l8_data.rio.crs)
print(l8_data.rio.nodata)
print(l8_data.rio.bounds())
print(l8_data.rio.width)
print(l8_data.rio.height)
print(l8_data.rio.crs.wkt)


# In[ ]:


l8_data.values
#l8_data


# In[ ]:


l8_data[4].plot() #plotting one band from the training imagery


# In[ ]:


#from 3D ----> 2D
l8_data = l8_data[0, :, :]
print(l8_data.shape)


# In[ ]:


from numpy import savetxt

# save to csv file
savetxt('/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/Landsat_Satellite_Data_Training.csv', l8_data, delimiter=',')


# In[ ]:


l8_data



def write_image(arr: np.array, save_path: os.PathLike, **raster_meta) -> None:
    """
    Write a Geotiff to disk with the given raster metadata.
    Convenience function that automatically sets permissions correctly.
    Args:
        arr (np.array): The data to write to disk in geotif format
        save_path (os.PathLike): The path to write to
    """
    with rasterio.open(save_path, "w", **raster_meta) as target:
        target.write(arr)
      # Allow group workspace users access


def downsample_image(
    image: rasterio.DatasetReader,
    bands: List[int],
    downsample_factor: int = 10.311,
    resampling: Resampling = Resampling.bilinear,
) -> Tuple[np.ndarray, Any]:
    """
    Downsample the given bands of a raster image by the given downsample_fator.
    Args:
        image (rasterio.DatasetReader): Rasterio IO handle to the image
        bands (List[int]): The bands to downsample
        downsample_factor (int, optional): Factor by which the image will be
            downsampled. Defaults to 2.
        resampling (Resampling, optional): Resampling algorithm to use. Must be one of
            Rasterio's built-in resampling algorithms. Defaults to Resampling.bilinear.
    Returns:
        Tuple[np.ndarray, Any]: Return the resampled bands of the image as a numpy
            array together with the transform.
    """

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
    """
    Generate downsample of the raster file at `file_path` and save it in same folder.
    Saved file will have appendix `_downsample_{downsample_factor}x.tif`
    Args:
        file_path (os.PathLike): The path to the raster image todownsample
        downsample_factor (int, optional): The downsampling factor to use.
            Defaults to 2.
        resampling (Resampling, optional): The resampling algorithm to use.
            Defaults to Resampling.bilinear.
        overwrite (bool, optional): Iff True, any existing downsampling file with the
            same downsampling factor will be overwritten. Defaults to False.
    """

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


# In[ ]:


generate_downsample(l8_reprojected)


# In[ ]:


#visualising resampled landsat imagery
l8_resampled = '/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/L8_training_data/resampled_satellite_nans.tif'
l8_resampled = xr.open_rasterio(l8_resampled)
print(l8_resampled)


# In[ ]:


l8_resampled[5].plot()


# In[ ]:


l8_resampled.values



### turning all 0 non data vlaues into nans
#l8_final = l8_resampled.where(l8_resampled !=0)
l8_filled = l8_resampled.fillna(0)
l8_filled.fillna(0)


# In[ ]:


#converting to raster for use in classification
l8_filled.rio.to_raster('/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/classification_training_data/final_filled_l8_training_data.tif')


