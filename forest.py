# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 13:23:11 2020

@author: Ranjit
"""

# 1.0 Clear memory
%reset -f

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
import os
from sklearn.preprocessing import OneHotEncoder
# Read data
pd.options.display.max_columns =200
os.chdir("D:\\Python\\7-Mar\\01032020\\forest_covtype")
os.listdir()
forest = pd.read_csv("train.csv.zip")
forest.head()
forest.columns
forest.pop('Id')

forest.dtypes.value_counts()
np.max(forest).max()
np.min(forest).min()

y = forest.pop('Cover_Type')
X =forest;
X = X.astype('int32')
X.dtypes
X.isnull().sum().sum()
X.nunique()
X.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace = True)
num_columns = X.columns[:10]
cat_columns = X.columns[10:]
X.shape
ss = StandardScaler()
ohe = OneHotEncoder()
num_scaled = ss.fit_transform(X.loc[:, num_columns])
cat_scaled = ohe.fit_transform(X.loc[:, cat_columns])

from sklearn.compose import ColumnTransformer
op1 = ('num', ss, num_columns)
op2 = ('cat', ohe, cat_columns)
ct = ColumnTransformer([op1, op2])

data = ct.fit_transform(X)
data.shape

from sklearn.preprocessing import KBinsDiscretizer

kb1 = KBinsDiscretizer(n_bins=7, strategy = 'kmeans')
dt = kb1.fit_transform(num_scaled) #can also use data[:, :10]    
dtt = dt.todense()
dx = np.hstack([dtt,data])

dx.shape
from sklearn.model_selection import train_test_split      
X_train, X_test, y_train, y_test = train_test_split(
                                                    dx,
                                                    y,
                                                    test_size = 0.25)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
np.sum(y_test == y_pred)/ len(y_pred)
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_test, y_pred)

from sklearn.decomposition import PCA
pca = PCA()
dx1 = pca.fit_transform(dx)
dx1.shape
np.std(dx1[:, 0])
np.std(dx1[:, 1])
np.std(dx1[:, 2])
np.std(dx1[:, 3])
np.std(dx1[:, -1])

np.std(dx[:, 0])
np.std(dx[:, 1])
np.std(dx[:, 2])
np.std(dx[:, 3])
np.sum(pca.explained_variance_ratio_)
pca.explained_variance_

import seaborn as sns
sns.lineplot(x = range(164), y= pca.explained_variance_ratio_)

