#!/usr/bin/env python
# coding: utf-8

# ### Imports

# Code adapted from: https://towardsdatascience.com/land-cover-classification-in-satellite-imagery-using-python-ae39dbf2929

# In[10]:


# Import Python 3's print function and division
from __future__ import print_function, division

# Import GDAL, NumPy, and matplotlib
from osgeo import gdal, gdal_array
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

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
from sklearn.svm import NuSVC




from sklearn.svm import SVC


# ### Prepare Data

# In[4]:


# Read in our satellite and label image
satellite_img = gdal.Open('/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/classification_training_data/final_filled_l8_training_data.tif', gdal.GA_ReadOnly)
training_img = gdal.Open('/gws/nopw/j04/ai4er/users/jl2182/data/Mres_Data/classification_training_data/third_remerge_landcover_training_data.tif', gdal.GA_ReadOnly)


# In[5]:



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


# In[6]:


unique, counts = np.unique(lbls, return_counts=True)
list(zip(unique, counts))


# In[7]:


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


# ## Stratify Data

# k-fold

# In[6]:


#stratified k-cross validation to balance the classes
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(X, y)


# In[7]:


StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# shuffle split

# In[8]:


from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

StratifiedShuffleSplit(n_splits=10, random_state=0)
        
for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# ### Train SVM

# In[15]:


#This estimator implements regularized linear models with stochastic gradient descent (SGD) learning: the gradient of 
#the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength 
#schedule (aka learning rate). SGD allows minibatch (online/out-of-core) learning via the partial_fit method. 
#The loss function defaults to ‘hinge’, which gives a linear SVM.

#tol is the stopping criterion. If it is not None, training will stop when (loss > best_loss - tol) for n_iter_no_change 
#consecutive epochs. Convergence is checked against the training loss or the validation loss depending on the 
#early_stopping parameter.

#learning rate is set to optimal where ‘optimal’: eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a 
#heuristic proposed by Leon Bottou. 
clf = NuSVC(nu=0.1)


# In[16]:


# Fit Data
train = clf.fit(X_train, y_train)


# In[10]:


print('The training accuracy is: {score}%'.format(score= train.score(X_train, y_train) * 100))


# ## Training Performance

# In[11]:


# Predict labels for test data
svm_pred = svm.predict(X_test)


# In[12]:


target_names = ['Cropland', 'Shrubland', 'Forest', 'Urban', 'Water']


# In[13]:


# Accuracy and Classification Reeport
print(f"Accuracy: {accuracy_score(y_test, svm_pred)*100}")
print(classification_report(y_test, svm_pred, target_names=target_names))


# In[14]:


disp = metrics.plot_confusion_matrix(svm, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()


# In[ ]:




