Python 3.8.1 (tags/v3.8.1:1b293b6, Dec 18 2019, 23:11:46) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandas as pd
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from sklearn.linear_model import LinearRegression
>>> from sklearn.preprocessing import PolynomialFeatures
>>> df = pd.read_csv("C:\\Users\shashikant\Desktop\polynomial_regression\polynomial.csv")
>>> x = df[['level']].values
>>> y = df[['salary']].values
>>> model = LinearRegression()
>>> model.fit(x,y)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
>>> poly = PolynomialFeatures()
>>> p_x = poly.fit_transform(x)
>>> poly.fit(p_x,y)
PolynomialFeatures(degree=2, include_bias=True, interaction_only=False,
                   order='C')
>>> model1 = LinearRegression()
>>> model1.fit(p_x,y)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
>>> plt.title('Linear Model')
Text(0.5, 1.0, 'Linear Model')
>>> plt.xlabel('Position Level')
Text(0.5, 0, 'Position Level')
>>> plt.ylabel('salary')
Text(0, 0.5, 'salary')
>>> plt.scatter(x,y,color = 'r')
<matplotlib.collections.PathCollection object at 0x0000000018C33F70>
>>> plt.plot(x,model.predict(x),color = 'b')
[<matplotlib.lines.Line2D object at 0x0000000018C4E4C0>]
>>> plt.show()
>>> plt.scatter(df.level,df.salary,color = "r")
<matplotlib.collections.PathCollection object at 0x0000000018E745E0>
>>> plt.plot(x,model1.predict(poly.fit_transform(x)),color = 'b')
[<matplotlib.lines.Line2D object at 0x0000000018E74C10>]
>>> plt.title('Polynomial Regression Model')
Text(0.5, 1.0, 'Polynomial Regression Model')
>>> plt.xlabel('Position Label')
Text(0.5, 0, 'Position Label')
>>> plt.ylabel('Salary')
Text(0, 0.5, 'Salary')
>>> plt.show()
>>> 