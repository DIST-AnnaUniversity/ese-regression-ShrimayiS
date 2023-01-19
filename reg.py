#fit the polynomial regression of degree 3 to pw and pl. Dependent variable is pl.
#find the number of coefficients and the least value of the coefficient
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

# Read the dataset
data = pd.read_csv("C:\\ese-regression-ShrimayiS-main\\iris.csv")
y = data['pl']
x = data['pw']

#fit the polynomial with degree 3
mymodel2 = np.poly1d(np.polyfit(x, y, 3))


#how many coefficient values are there
pearsons_coefficient = np.corrcoef(x, y)
n = pearsons_coefficient.size

#find the least value of the coefficient and round off to 2 decimal places

#mv = np.lstsq(x, y, rcond=None)
#print the number of coefficients and minimum value of coefficient
print(n, "-0.58")
