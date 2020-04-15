# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:52:32 2020

@author: Elif
"""

import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("polynomialRgression.csv", sep=";")

x=df.max_speed.values.reshape(-1,1)
y=df.car_price.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("max_speed")
plt.ylabel("car_price")
plt.show()


#inear regression  y = b0 + b1*x
#multiple linear regression  y = b0 + b1*x1 + b2*x2
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x,y)

#predict
y_head=lr.predict(x)
plt.plot(x,y_head,color="red",label="linear")
plt.show()

print("10.000 cost:",lr.predict([[10000]]))

#polynomial regresion y=b0+b1*x+b2*x^2..+bn*x^n

from sklearn.preprocessing import PolynomialFeatures
#polynomial_regression=PolynomialFeatures(degree=2)
polynomial_regression=PolynomialFeatures(degree=4)
x_polynomial=polynomial_regression.fit_transform(x)

#fit
linear_regression2=LinearRegression()
linear_regression2.fit(x_polynomial,y)

##vizulation

y_head2=linear_regression2.predict(x_polynomial)

plt.plot(x,y_head2,color="green",label="poly")
plt.legend()
plt.show()












