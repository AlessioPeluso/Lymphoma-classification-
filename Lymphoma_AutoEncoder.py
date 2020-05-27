
# --------------------------------------------------------------------------------------------------

# Libraries
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import random as rd
import keras

from numpy.random import seed
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.layers import Input, add
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers
from keras.regularizers import l2
from keras.utils import np_utils
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from scipy import stats
from pylab import rcParams

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------------------------------

# Read data
data = pd.read_csv('dlbcl_preprocessed.txt', sep=" ", header=None)

print(data.shape)

data_t = data.T

# create header var
header = data_t.iloc[0]

# Replace the dataframe with a new one which does not contain the first row
data_t = data_t[1:]
# Rename the dataframe's column values with the header variable
data_t.rename(columns = header)

# --------------------------------------------------------------------------------------------------

# Cleaning Dataset

# replace
data.y.replace((0, 1), ('normal', 'tumor'), inplace=True)

# data rearrange
Y = data_t.iloc[:,2647] #58-19
X = data_t.iloc[:,0:2646]

X.shape, Y.shape

# --------------------------------------------------------------------------------------------------

# Principal Component Analysis

from sklearn.preprocessing import StandardScaler
# Standardizing the features
X = StandardScaler().fit_transform(X)

pca = PCA(n_components = 77)
X_r = pca.fit(X.T)

print(pca.explained_variance_ratio_[1:30])
pca.components_.shape

per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

# plot of percentage of explained variance
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.axvline(x=41, color='g', linestyle ='dashed',linewidth=1)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

cumsum_ = np.cumsum(pca.explained_variance_ratio_)
r = range(0,77)

# how many features for the 90% percentile
plt.plot(r,cumsum_)
plt.axvline(x=41, color='g', linestyle ='dashed',linewidth=1)
plt.axhline(y=0.9, color='g', linestyle ='dashed',linewidth=1)

# Determine which genes had the biggest influence on PC1

# get the name of the top 10 measurements (genes) that contribute
# most to pc1.
# first, get the loading scores
loading_scores = pd.Series(pca.components_[0])
# now sort the loading scores based on their magnitude
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

# get the names of the top 10 genes
top_10_genes = sorted_loading_scores[0:10].index.values

# print the gene names and their scores (and +/- sign)
print(loading_scores[top_10_genes], '\n', header[[12,7,28,15,20,8,41,1,17,91]])


# PCA Classification with Neural Network

# NNet for classifier w\ softmax
seed = 23
np.random.seed(seed)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(77, input_dim=77, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# 1 evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, pca.components_, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# Results: 69.05% (9.86%)

# 2 larger model
def create_larger():
	# create model
	model = Sequential()
	model.add(Dense(77, input_dim=77, kernel_initializer='normal', activation='relu'))
	model.add(Dense(35, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_larger, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, pca.components_, encoded_Y, cv=kfold)
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# Larger: 71.73% (10.34%)

# 3 smaller model
def create_smaller():
	# create model
	model = Sequential()
	model.add(Dense(35, input_dim=77, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, pca.components_, encoded_Y, cv=kfold)
print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# Smaller: 72.98% (5.95%)


# --------------------------------------------------------------------------------------------------

# Sparse Autoencoder

Y = data_t.iloc[:,2647]

n = int(data_t.shape[1]*0.1)
s = np.random.randint(1,2646, n)

df_ = pd.DataFrame(data_, index=list(range(0, 77))) #77x77
df_raw = pd.DataFrame(data_t.iloc[:,s], index=list(range(1,77))) #76x264

X = pd.concat([df_.iloc[1:77,0:76], df_raw], axis=1, join_axes=[df_.iloc[1:102,0:101].index]) #76x340

Xs = minmax_scale(X, axis = 0)
ncol = Xs.shape[1]

Xs.shape,ncol

# TRAIN e TEST
S_X_train, S_X_test, S_Y_train, S_Y_test = train_test_split(Xs, Y[1:77], test_size = 0.4)
S_X_test.shape, S_X_train.shape

#------------------
#Sparse Autoencoder
#------------------
# from keras import optimizers
# # All parameter gradients will be clipped to a maximum norm of 1.
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)

input_dim = Input(shape = (ncol, ))
encoding_dim = 75
encoded = Dense(encoding_dim, activation = 'sigmoid',activity_regularizer=regularizers.l1(10e-5))(input_dim)
decoded = Dense(ncol, activation = 'sigmoid')(encoded)
autoencoder = Model(inputs = input_dim, outputs = decoded)
autoencoder.compile(optimizer = 'adam', loss = 'mse')
history = autoencoder.fit(S_X_train, S_X_train, epochs = 1000, batch_size = 15, shuffle = True, validation_data = (S_X_test, S_X_test), verbose=0)

# THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
encoder = Model(inputs = input_dim, outputs = encoded)
encoded_input = Input(shape = (encoding_dim, ))
encoded_out = encoder.predict(S_X_test)
encoded_out2 = encoder.predict(S_X_train)
result = encoder.predict(Xs)

#print shape
encoded_out.shape, encoded_out2.shape

# Plot all losses
print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# Classification

# NNet for classifier w\ softmax
seed = 23
np.random.seed(seed)

S_Y = Y.iloc[1:77]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(S_Y)
encoded_Y = encoder.transform(S_Y)

def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(75, input_dim=75, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# 1 evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, result, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# Results: 90.00% (9.35%)

# 2 larger model
def create_larger():
	# create model
	model = Sequential()
	model.add(Dense(75, input_dim=75, kernel_initializer='normal', activation='relu'))
	model.add(Dense(35, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_larger, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, result, encoded_Y, cv=kfold)
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# Larger: 92.50% (8.29%)

# 3 smaller model
def create_smaller():
	# create model
	model = Sequential()
	model.add(Dense(35, input_dim=75, kernel_initializer='normal', activation='softmax'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, result, encoded_Y, cv=kfold)
print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# Smaller: 85.65% (10.39%)
