# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:20:08 2020

@author: Elif
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("randomforestdata.csv",sep=";", header=None)

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100,random_state=42)
## 100 tane tree ve 42 render için aynı random değerler

rf.fit(x,y)

##print("7.8 seviyesinde fiaytın ne kadar olduğu:",rf.predict([[7.8]]))

y_head=rf.predict(x)

from sklearn.metrics import r2_score
print("r_score:",r2_score(y,y_head))
plt.plot(x,y_head,color="red")
