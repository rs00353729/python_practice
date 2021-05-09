# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 11:31:32 2020

@author: Ranjit
"""

import numpy as np
import pandas as pd
# 1.2 For plotting
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl     # For creating colormaps
import seaborn as sns
# 1.3 For data processing
from sklearn.preprocessing import StandardScaler as ss
# 1.4 OS related
import os

### Plotly for interactive plots 'conda install -c plotly plotly=4.5.1
import plotly.express as px
import plotly.graph_objects as go
%matplotlib inline
# 1.5 Go to folder containing data file
#os.chdir("D:\\data\\OneDrive\\Documents\\advertising")

os.chdir("D:\\Python\\22-Feb")
os.listdir()            # List all files in the folder

# 1.6 Read file and while reading file,
#      convert 'Timestamp' to datetime time
ad = pd.read_csv("Border_Crossing_Entry_Data.csv");
ad.dtypes
ad['date_dt'] = pd.to_datetime(ad['Date'])
ad['Year'] = ad['date_dt'].dt.year
ad['Month'] = ad['date_dt'].dt.month
ad['weekday']=ad['date_dt'].dt.weekday
pd.set_option('display.max_columns', 500)

ad.head()

ad.columns

type(ad)
ad['Border']['Value'].nunique()

ad['Year'].value_counts().sort_index().plot(kind='bar')
bordergroup=ad[['Border','Value']].groupby('Border').sum()
values=bordergroup.values.flatten()
labels=bordergroup.index
bordergroup.plot.pie(y='Value')
ad['Year'].nunique()
ad.groupby(['State'])['Year'].count()
ad.groupby(['State']).count()
ad.groupby(['State'])['Year'].count()

p=ad[['Year','Border','Value']].set_index('Year')

p = p.groupby(['Year','Border']).sum()

val_mex= p.loc(axis=0)[:,'US-Mexico Border'].values.flatten().tolist()
val_can= p.loc(axis=0)[:,'US-Canada Border'].values.flatten().tolist()
yrs=p.unstack(level=1).index.values

fig = go.Figure(go.Bar(x = yrs, y = val_mex, name='US-Mexico Border'))
fig.add_trace(go.Bar(x = yrs, y = val_can, name='US-Canada Border'))
fig.update_layout(title = 'Border & year wise valume pattern', barmode='stack', xaxis={'categoryorder':'category ascending'})
fig.show()
                      #   and status of nulls
ad.shape                # (1000, 10)
ad.columns.values
len(ad.columns)         # 10 attributes

# 1.9 Categorical data value counts
#     Or number of levels per category
len(ad.City.unique())                   # 969 cities out of 1000
ad.City.value_counts()


state_grp = ad.groupby(['State'])
state_grp.count()