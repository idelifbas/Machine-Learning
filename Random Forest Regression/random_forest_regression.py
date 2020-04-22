# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:41:07 2020

@author: Elif
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("randomforestdata.csv",sep=";", header= None)

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

##random forest regression
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(x,y)

print("7.8 level's price':",rf.predict([[7.8]]))

x_=np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head=rf.predict(x_)

#visualze
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("level")
plt.ylabel("price")
plt.show()







