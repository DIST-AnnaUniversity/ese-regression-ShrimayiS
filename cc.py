#In the given dataset, find the correlation coefficient between pw and pl and round to 2 decimal places
import pandas as pd
import numpy as np

# Read the dataset
data = pd.read_csv("iris.csv")

# Take dependent variable as 'pl' and independent variable as 'pw'
y = data['pl']
x = data['pw']
pearsons_coefficient = np.corrcoef(x, y)



# Print the correlation coefficient and round off to 2 decimal places
print(np.round(pearsons_coefficient,decimals = 2, out=None))
