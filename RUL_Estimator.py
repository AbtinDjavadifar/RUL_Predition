# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import featuretools as ft
import utils
import os
from utils import relative_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
     
# =============================================================================
# Loading training data
# =============================================================================
data = utils.load_data('train_FD004.txt')
#data = utils.load_data('train_FD004_7v.txt')
#data.head()

#%%
# =============================================================================
# Creating cutoff times
# =============================================================================
cutoff_times = utils.make_cutoff_times(data)
#cutoff_times.head()

#%%
# =============================================================================
# Making entitysets
# =============================================================================
def make_entityset(data):
    es = ft.EntitySet('Dataset')
    es.entity_from_dataframe(dataframe=data,
                             entity_id='recordings',
                             index='index',
                             time_index='time')

    es.normalize_entity(base_entity_id='recordings', 
                        new_entity_id='engines',
                        index='engine_no')

    es.normalize_entity(base_entity_id='recordings', 
                        new_entity_id='cycles',
                        index='time_in_cycles')
    return es
es = make_entityset(data)

#es
#es["recordings"].variables
#es["engines"].variables
#es["cycles"].variables
#es["recordings"].df.head(5)
#es["engines"].df.head(5)
#es["cycles"].df.head(5)
es.plot()

#%%
# =============================================================================
# Creating features
# =============================================================================
fm, features = ft.dfs(entityset=es, 
                      target_entity='engines',
                      agg_primitives=['last', 'max', 'min'],
#                      agg_primitives=['last', 'max', 'min', 'mean', 'std'],
                      trans_primitives=[],
                      cutoff_time=cutoff_times,
                      max_depth=3,
                      verbose=True)
fm.to_csv('simple_fm.csv')

#%%
# =============================================================================
# Splitting training data
# =============================================================================
fm = pd.read_csv('simple_fm.csv', index_col='engine_no')
X_train = fm.copy().fillna(0)
y_train = X_train.pop('RUL')

X_training, X_validating, y_training, y_validating = train_test_split(X_train, y_train, random_state=17)

#%%
# =============================================================================
# Prediction using median baseline 1 in training data
# =============================================================================
medianpredict1 = [np.median(y_training) for _ in y_validating]
print('Baseline by median label (training data): Mean Abs Error = {:.2f}'.format(
    mean_absolute_error(medianpredict1, y_validating)))
print('Baseline by median label (training data): Root Mean Square Error = {:.2f}'.format(np.sqrt(mean_squared_error(y_validating, medianpredict1))))
print('Baseline by median label (training data): Relative Error = {:.2f}'.format(relative_error(y_validating.values, medianpredict1)))

#%%
# =============================================================================
# # Prediction using median baseline 2 in training data
# =============================================================================
#takes the rows that their engine id is selected for training
recordings_from_train = es['recordings'].df[es['recordings'].df['engine_no'].isin(y_training.index)]
#recordings_from_train.groupby(['engine_no']).apply(lambda df: df.shape[0]): replaces the data of each row by number of cycles of that engine  
median_life = np.median(recordings_from_train.groupby(['engine_no']).apply(lambda df: df.shape[0]))

#takes the rows that their engine id is selected for testing
recordings_from_test = es['recordings'].df[es['recordings'].df['engine_no'].isin(y_validating.index)]
#number of cycles for the engine - RUL
life_in_test = recordings_from_test.groupby(['engine_no']).apply(lambda df: df.shape[0])-y_validating

medianpredict2 = (median_life - life_in_test).apply(lambda row: max(row, 0))
print('Baseline by median life (training data): Mean Abs Error = {:.2f}'.format(
    mean_absolute_error(medianpredict2, y_validating)))
print('Baseline by median life (training data): Root Mean Square Error = {:.2f}'.format(np.sqrt(mean_squared_error(y_validating, medianpredict2))))
print('Baseline by median life (training data): Relative Error = {:.2f}'.format(relative_error(y_validating.values, medianpredict2)))

#%%
# =============================================================================
# Prediction using RFR in training data
# =============================================================================
RFRreg = RandomForestRegressor(n_estimators=100)
RFRreg.fit(X_training, y_training)
    
RFRpreds = RFRreg.predict(X_validating)
print('RFR Mean Abs Error (training data): {:.2f}'.format(mean_absolute_error(RFRpreds, y_validating)))
print('RFR Root Mean Square Error (training data): {:.2f}'.format(np.sqrt(mean_squared_error(y_validating, RFRpreds))))
print('RFR Relative Error (training data): {:.2f}'.format(relative_error(y_validating.values, RFRpreds)))
#high_imp_feats = utils.feature_importances(X, RFRreg, feats=10)

#%%
# =============================================================================
# Loading test data
# =============================================================================
data2 = utils.load_data('test_FD004.txt')
#data2 = utils.load_data('test_FD004_7v.txt')
es2 = make_entityset(data2)
fm2 = ft.calculate_feature_matrix(entityset=es2, features=features, verbose=True)
X_test = fm2.copy().fillna(0)
y_test = pd.read_csv('RUL_FD004.txt', sep=' ', header=-1, names=['RUL'], index_col=False)
#fm2.head()

#%%
# =============================================================================
# Prediction using RFR in test data
# =============================================================================
RFRreg.fit(X_train, y_train)

RFRpreds2 = RFRreg.predict(X_test)
print('RFR Mean Abs Error (test data): {:.2f}'.format(mean_absolute_error(RFRpreds2, y_test)))
print('RFR Root Mean Square Error (test data): {:.2f}'.format(np.sqrt(mean_squared_error(y_test, RFRpreds2))))
print('RFR Relative Error (test data): {:.2f}'.format(relative_error(y_test.values, RFRpreds2)))

#%%
# =============================================================================
# Prediction using median baseline 1 & 2 in test data
# =============================================================================
medianpredict1 = [np.median(y_training) for _ in RFRpreds2]
print('Baseline by median label (test data): Mean Abs Error = {:.2f}'.format(
    mean_absolute_error(medianpredict1, y_test)))
print('Baseline by median label (test data): Root Mean Square Error = {:.2f}'.format(np.sqrt(mean_squared_error(y_test, medianpredict1))))
print('Baseline by median label (test data): Relative Error = {:.2f}'.format(relative_error(y_test.values, medianpredict1)))

medianpredict2 = (median_life - es2['recordings'].df.groupby(['engine_no']).apply(lambda df: df.shape[0])).apply(lambda row: max(row, 0))
print('Baseline by median life (test data): Mean Abs Error = {:.2f}'.format(
    mean_absolute_error(medianpredict2, y_test)))
print('Baseline by median life (test data): Root Mean Square Error = {:.2f}'.format(np.sqrt(mean_squared_error(y_test, medianpredict2))))
print('Baseline by median life (test data): Relative Error = {:.2f}'.format(relative_error(y_test.values, medianpredict2.values)))

#%%
# =============================================================================
# Prediction using SVR in test data
# =============================================================================
SVRreg=SVR(kernel='rbf',epsilon=3.0,degree=3)
SVRreg.fit(X_train,y_train)

SVRpreds=SVRreg.predict(X_test)
print('SVR Mean Abs Error (test data): {:.2f}'.format(mean_absolute_error(SVRpreds, y_test)))
print('SVR Root Mean Square Error (test data): {:.2f}'.format(np.sqrt(mean_squared_error(y_test, SVRpreds))))
print('SVR Relative Error (test data): {:.2f}'.format(relative_error(y_test.values, SVRpreds)))

#%%
# =============================================================================
#  Prediction using MLP in test data
# =============================================================================
MLPreg=MLPRegressor(hidden_layer_sizes=(2, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, max_iter=4000, tol=0.0001, momentum=0.9, epsilon=1e-08)
MLPreg.fit(X_train,y_train)

MLPpreds=MLPreg.predict(X_test)
print('MLP Mean Abs Error (test data): {:.2f}'.format(mean_absolute_error(MLPpreds, y_test)))
print('MLP Root Mean Square Error (test data): {:.2f}'.format(np.sqrt(mean_squared_error(y_test, MLPpreds))))
print('MLP Relative Error (test data): {:.2f}'.format(relative_error(y_test.values, MLPpreds)))

#%%
# =============================================================================
#  Prediction using CART in test data
# =============================================================================
CARTreg = tree.DecisionTreeRegressor()
CARTreg.fit(X_train,y_train)

CARTpreds = CARTreg.predict(X_test)
print('CART Mean Abs Error (test data): {:.2f}'.format(mean_absolute_error(CARTpreds, y_test)))
print('CART Root Mean Square Error (test data): {:.2f}'.format(np.sqrt(mean_squared_error(y_test, CARTpreds))))
print('CART Relative Error (test data): {:.2f}'.format(relative_error(y_test.values, CARTpreds)))

#%%
# =============================================================================
# Saving output files
# =============================================================================
try:
    os.mkdir("output")
except:
    pass

fm.to_csv('output/simple_train_feature_matrix.csv')
cutoff_times.to_csv('output/simple_train_label_times.csv')
fm2.to_csv('output/simple_test_feature_matrix.csv')
