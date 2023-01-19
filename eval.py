#Evaluate the polynomial regression through r2 score with degree 2 and degree 3. Compare both.
#If same, print "same" (round off the r2 score to 2 decimal places)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

# Read the dataset
data = pd.read_csv("C:\\ese-regression-ShrimayiS-main\\iris.csv")
y = data['pl']
x = data['pw']

#fit the polynomial with degree 2 and 3
mymodel = np.poly1d(np.polyfit(x, y, 2))
mymodel2 = np.poly1d(np.polyfit(x, y, 3))



#evaluate both polynomials with r2 score and round off to 2 decimals

n1 =round((r2_score(y, mymodel(x))),2)
#print(n1)
n2 =round((r2_score(y, mymodel2(x))),2)
#print(n2)
if n1 == n2:
    print("same")
