# -*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from numpy.linalg import inv 
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Loading data
# =============================================================================
f = open('train_FD004.txt', 'r')
x_train = np.array([[float(num) for num in line.split(' ') if num!='\n' and num!=''] for line in f])
f.close()
#print(np.shape(x_train))

#%%
# =============================================================================
# Computing HIs with 21 variables
# =============================================================================
num_traj=int(x_train[-1][0])
N0=0.9
y_train=np.empty((0,1))
H=np.empty((0,1))
#y_train=np.matrix([[np.ones((int(np.shape(x_train)[0] * N), 1))],[np.zeros(((np.shape(x_train)[0] - int(np.shape(x_train)[0] * N)), 1))]])
for i in range(num_traj):
    i+=1
    count=np.sum(x_train[:,0] == i)
    place=np.sum(x_train[:,0] <= i)
    F=x_train[(place-count):place , 5:]
    Soff=np.empty((0,1))
    y_train=np.append(y_train, np.zeros((int(count * N0), 1)), axis=0)
    y_train=np.append(y_train, np.ones((count - int(count * N0), 1)), axis=0)
    Soff=np.append(Soff, np.zeros((int(count * N0), 1)), axis=0)
    Soff=np.append(Soff, np.ones((count - int(count * N0), 1)), axis=0)
    T = np.matmul(np.matmul(inv(np.matmul(F.transpose() , F)) , F.transpose()) , Soff)
    H=np.append(H, np.matmul(F , T), axis=0)

#y_train=H
    
#%%
# =============================================================================
# Fitting RFs
# =============================================================================
X = x_train[:, 5:]
Y = y_train
names = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
rf = RandomForestRegressor()
rf.fit(X, Y)
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True))

#%%
# =============================================================================
# plotting variable improtance diagram with 21 variables
# =============================================================================
index = np.arange(len(names))
#plt.figure()
df = pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),reverse=True))
df.plot.bar(legend=None)
#plt.bar(names, rf.feature_importances_)
plt.xlabel('Variables', fontsize=18)
plt.ylabel('Variable importance', fontsize=18)
plt.xticks(index, np.array(df.iloc[:, 1]), fontsize=15, rotation=15)
plt.title('Variable importance of 21 variables', fontsize=20)
plt.show()

#%%
# =============================================================================
# plotting health indices diagram with 21 variables
# =============================================================================
plt.figure()
plt.plot(x_train[:, 1], H)
plt.xlim(0, 600)
plt.ylim(-0.5, 1.5)
plt.xlabel('RUL (Number of Cycles)', fontsize=15)
plt.ylabel('HI', fontsize=15)
ax = plt.gca()
ax.invert_xaxis()

#%%
# =============================================================================
# Finding first 7 important variables
# =============================================================================
imp_var_indx= rf.feature_importances_.argsort()[-7:][::-1] + 5
indices=np.sort(np.append(imp_var_indx, [0,1,2,3,4]))
x_train_7v=x_train[:, indices]

#%%
# =============================================================================
# Computing HIs with 7 variables
# =============================================================================
num_traj_7v=int(x_train_7v[-1][0])
N0=0.9
y_train_7v=np.empty((0,1))
H_7v=np.empty((0,1))

for i in range(num_traj_7v):
    i+=1
    count=np.sum(x_train_7v[:,0] == i)
    place=np.sum(x_train_7v[:,0] <= i)
    F_7v=x_train_7v[(place-count):place , 5:]
    Soff_7v=np.empty((0,1))
    y_train_7v=np.append(y_train_7v, np.zeros((int(count * N0), 1)), axis=0)
    y_train_7v=np.append(y_train_7v, np.ones((count - int(count * N0), 1)), axis=0)
    Soff_7v=np.append(Soff_7v, np.zeros((int(count * N0), 1)), axis=0)
    Soff_7v=np.append(Soff_7v, np.ones((count - int(count * N0), 1)), axis=0)
    T_7v = np.matmul(np.matmul(inv(np.matmul(F_7v.transpose() , F_7v)) , F_7v.transpose()) , Soff_7v)
    H_7v=np.append(H_7v, np.matmul(F_7v , T_7v), axis=0)

#%%
# =============================================================================
# plotting health indices diagram with 7 variables
# =============================================================================
plt.figure()
plt.plot(x_train_7v[:, 1], H_7v)
plt.xlim(0, 600)
plt.ylim(-0.5, 1.5)
plt.xlabel('RUL (Number of Cycles)', fontsize=15)
plt.ylabel('HI', fontsize=15)
ax = plt.gca()
ax.invert_xaxis()

#%%
# =============================================================================
# Saving training and test arrays with 7 variable in text file
# =============================================================================
np.savetxt("train_FD004_7v.txt",x_train_7v , fmt="%s")

f = open('test_FD004.txt', 'r')
x_test = np.array([[float(num) for num in line.split(' ') if num!='\n' and num!=''] for line in f])
f.close()
x_test_7v=x_test[:, indices]
np.savetxt("test_FD004_7v.txt",x_test_7v , fmt="%s")
