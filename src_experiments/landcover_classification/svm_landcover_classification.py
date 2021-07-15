#!/usr/bin/env python
# coding: utf-8

# ### Imports

# Code adapted from: https://towardsdatascience.com/land-cover-classification-in-satellite-imagery-using-python-ae39dbf2929

# In[1]:


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
from pprint import pprint
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC


# ### Prepare Data

# In[2]:


# Read in our satellite and label image
satellite_img = gdal.Open('/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/classification_training_data/final_filled_l8_training_data.tif', gdal.GA_ReadOnly)
training_img = gdal.Open('/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/classification_training_data/third_remerge_landcover_training_data.tif', gdal.GA_ReadOnly)


# In[3]:



img = np.zeros((satellite_img.RasterYSize, satellite_img.RasterXSize, satellite_img.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(satellite_img.GetRasterBand(1).DataType))
for b in range(img.shape[2]):
    img[:, :, b] = satellite_img.GetRasterBand(b + 1).ReadAsArray()

lbls = training_img.GetRasterBand(1).ReadAsArray().astype(np.uint8)

# Display them
plt.subplot(121)
plt.imshow(img[:, :, 4], cmap=plt.cm.tab20b)
plt.title('SWIR1')

plt.subplot(122)
plt.imshow(lbls, cmap=plt.cm.terrain)
plt.title('Training Data')

plt.show()


# In[5]:


unique, counts = np.unique(lbls, return_counts=True)
list(zip(unique, counts))


# In[18]:


# Find how many non-zero entries we have -- i.e. how many training data samples?
n_samples = (lbls !=6).sum()
print('We have {n} samples'.format(n=n_samples))

# What are our classification labels?
labels = np.unique(lbls[lbls !=6])
print('The training data include {n} classes: {classes}'.format(n=labels.size,
                                                                classes=labels))
# We will need a "X" matrix containing our features, and a "y" array containing our labels
#     These will have n_samples rows
#     In other languages we would need to allocate these and them loop to fill them, but NumPy can be faster

#this is a quick numpy trick for flattening
X = img[lbls !=6]  # include 8th band, which is Fmask, for now
y = lbls[lbls !=6]


print('Our X matrix is sized: {sz}'.format(sz=X.shape))
print('Our y array is sized: {sz}'.format(sz=y.shape))


# ### Train SVM

# In[7]:


#stratified k-cross validation to balance the classes
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(X, y)


# In[8]:


StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[9]:


#This estimator implements regularized linear models with stochastic gradient descent (SGD) learning: the gradient of
#the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength
#schedule (aka learning rate). SGD allows minibatch (online/out-of-core) learning via the partial_fit method.
#The loss function defaults to ‘hinge’, which gives a linear SVM.

#tol is the stopping criterion. If it is not None, training will stop when (loss > best_loss - tol) for n_iter_no_change
#consecutive epochs. Convergence is checked against the training loss or the validation loss depending on the
#early_stopping parameter.

#learning rate is set to optimal where ‘optimal’: eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a
#heuristic proposed by Leon Bottou.
svm = SGDClassifier(max_iter = 1000,tol=1e-3)


### HYPERPARAMETER TESTING WITH RANDOM SEARCH

#Method 2: Code adapted from Hyperparameter Tuning the Random Forest in Python -
#https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

### Create the parameter grid
penalty = ['l1', 'l2', 'elasticnet']
alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive']
eta0 = [1, 10, 100]
# Create the random grid
param_distributions = dict(
penalty=penalty,
alpha=alpha,
learning_rate=learning_rate,
eta0=eta0)



### Instantiate the random search and fit
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
svm_random = RandomizedSearchCV(estimator = svm,  param_distributions=param_distributions, scoring='roc_auc',
verbose=1, n_jobs=-1, n_iter=1000)
random_result = svm_random.fit(X_train, y_train)
print('Best Score: ', random_result.best_score_)
print('Best Params: ', random_result.best_params_)
