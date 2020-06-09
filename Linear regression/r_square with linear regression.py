# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:42:16 2020

@author: Elif
"""

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("verisetim.csv",sep=";")

plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.show()

from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
x=df.deneyim.values.reshape(-1,1)
y=df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

y_head=linear_reg.predict(x)
plt.plot(x, y_head, color="red")

from sklearn.metrics import r2_score

print("r_square score:", r2_score(y,y_head))