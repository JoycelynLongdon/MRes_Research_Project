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
training_img = gdal.Open('/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/classification_training_data/final_landcover_training_data.tif', gdal.GA_ReadOnly)


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
n_samples = (lbls !=9).sum()
print('We have {n} samples'.format(n=n_samples))

#What are our classification labels?
labels = np.unique(lbls[lbls !=9])
print('The training data include {n} classes: {classes}'.format(n=labels.size,
                                                                classes=labels))
# We will need a "X" matrix containing our features, and a "y" array containing our labels
#     These will have n_samples rows
#     In other languages we would need to allocate these and them loop to fill them, but NumPy can be faster

#this is a quick numpy trick for flattening
X = img[lbls !=9]  # include 8th band, which is Fmask, for now
y = lbls[lbls !=9]


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



#current parameters in use
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())


### HYPERPARAMETER TESTING WITH GRID SEARCH
#Methood 1: Code adapted from "Dealing with multiclass data" -
# https://towardsdatascience.com/dealing-with-multiclass-data-78a1a27c5dcc

def hyper_param_rf_predict(X_train, y_train, X_test, y_test):
    #rfc = RandomForestClassifier()
    #rfc.fit(X_train, y_train)
    n_optimal_param_grid = {
    'bootstrap': [True],
    'max_depth': [20], #setting this so as not to create a tree that's too big
    #'max_features': [2, 3, 4, 10],
    'min_samples_leaf': [1],
    'min_samples_split': [2],
    'n_estimators': [500]
    }
    grid_search_optimal = GridSearchCV(estimator = rf, param_grid = n_optimal_param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)
    grid_search_optimal.fit(X_train, y_train)
    rf_pred_gs = grid_search_optimal.predict(X_test)
    print ("Random Forest Train Accuracy Baseline After Grid Search:", metrics.accuracy_score(y_train, grid_search_optimal.predict(X_train)))
    print ("Random Forest Test Accuracy Baseline After Grid Search:", metrics.accuracy_score(y_test, grid_search_optimal.predict(X_test)))
    print(confusion_matrix(y_test,rf_pred_gs))
    print(classification_report(y_test,rf_pred_gs))
    rf_train_acc = metrics.accuracy_score(y_train, rf.predict(X_train))
    rf_test_acc = metrics.accuracy_score(y_test, rf.predict(X_test))


    # plt.
    return(rf_train_acc, rf_test_accs)


hyper_param_rf_predict(X_train, y_train, X_test, y_test)


### HYPERPARAMETER TESTING WITH RANDOM SEARCH

#Method 2: Code adapted from Hyperparameter Tuning the Random Forest in Python -
#https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

### Create the parameter grid
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 500, num = 4)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 50, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)


### Instantiate the random search and fit
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)


# Note: The most important arguments in RandomizedSearchCV are n_iter, which controls the number of different combinations to try,
# and cv which is the number of folds to use for cross validation. More iterations will cover a wider search space and more cv folds
# reduces the chances of overfitting.


### Identify the best parameters
rf_random.best_params_


### Evaluate the performance of the grid search model to the baseline model
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

#baseline
base_model = rf
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)

#random grid search
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)


print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

