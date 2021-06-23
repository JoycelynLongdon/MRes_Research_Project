#!/usr/bin/env python
# coding: utf-8

# This code was adapted from Chris Holden (ceholden@gmail.com) Chaoter 5 Lesson on Ladncover Classification: https://ceholden.github.io/open-geo-tutorial/python/chapter_5_classification.html - https://github.com/ceholden
#
#

# ### IMPORTS

# Import Python 3's print function and division
from __future__ import print_function, division

# Import GDAL, NumPy, and matplotlib
from osgeo import gdal, gdal_array
import numpy as np
import matplotlib.pyplot as plt

## Sklearn Libraries
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc,             classification_report, recall_score, precision_recall_curve
from sklearn.metrics import accuracy_score
import pandas as pd
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV




### PREPARING THE DATASET

# Read in our satellite and label image
satellite_img = gdal.Open('/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/classification_training_data/final_filled_l8_training_data.tif', gdal.GA_ReadOnly)
training_img = gdal.Open('/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/classification_training_data/second_remerge_landcover_training_data.tif', gdal.GA_ReadOnly)


img = np.zeros((satellite_img.RasterYSize, satellite_img.RasterXSize, satellite_img.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(satellite_img.GetRasterBand(1).DataType))
for b in range(img.shape[2]):
    img[:, :, b] = satellite_img.GetRasterBand(b + 1).ReadAsArray()

lbls = training_img.GetRasterBand(1).ReadAsArray().astype(np.uint8)

# Display them
#plt.subplot(121)
#plt.imshow(img[:, :, 4], cmap=plt.cm.tab20b)
#plt.title('SWIR1')

#plt.subplot(122)
#plt.imshow(lbls, cmap=plt.cm.terrain)
#plt.title('Training Data')

#plt.show()



print(img.shape)
print(lbls.shape)


# CREATING THE X AND Y FEATURE AND LABEL MATRICES TO BE FED INTO THE RF


# Find how many non-zero entries we have -- i.e. how many training data samples?
n_samples = (lbls !=7).sum()
print('We have {n} samples'.format(n=n_samples))

#What are our classification labels?
labels = np.unique(lbls[lbls !=7])
print('The training data include {n} classes: {classes}'.format(n=labels.size,
                                                                classes=labels))
# We will need a "X" matrix containing our features, and a "y" array containing our labels
#     These will have n_samples rows
#     In other languages we would need to allocate these and them loop to fill them, but NumPy can be faster

#this is a quick numpy trick for flattening
X = img[lbls !=7]  # include 8th band, which is Fmask, for now
y = lbls[lbls !=7]


print('Our X matrix is sized: {sz}'.format(sz=X.shape))
print('Our y array is sized: {sz}'.format(sz=y.shape))


### TRAINING THE RANDOM FOREST

#stratified k-cross validation to balance the classes
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(X, y)

StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)


# Initialize our model with 500 trees
rf = RandomForestClassifier(n_estimators=500, oob_score=True)

# Fit our model to training data
train = rf.fit(X_train, y_train)


### TBASELINE RAINING PERFORMANCE

print('Our OOB prediction of our baseline model accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100))

# Setup a dataframe -- just like R
df = pd.DataFrame()
df['truth'] = y
df['predict'] = rf.predict(X)

# Cross-tabulate predictions
print(pd.crosstab(df['truth'], df['predict'], margins=True))


### BAESLINE VALIDATION PERFORMANCE

val = rf.predict(X_test)

target_names = ['Cropland', 'Mosaic Cropland', 'Mosaic Vegetation', 'Forest',
                         'Shrubland', 'Grassland', 'Urban', 'Water']

print(classification_report(y_test, val, target_names=target_names))


disp = metrics.plot_confusion_matrix(rf, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

#plt.show()


### BAND IMPORTANCE

bands = [1, 2, 3, 4, 5, 6,7,8,9,10]

for b, imp in zip(bands, rf.feature_importances_):
    print('Band {b} importance: {imp}'.format(b=b, imp=imp))

