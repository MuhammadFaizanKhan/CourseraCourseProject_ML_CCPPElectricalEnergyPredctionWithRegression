import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #matplotlib.pyplot (imported as plt) is a collection of command-style functions that make Matplotlib work like MATLAB.
import seaborn as sns #seaborn (imported as sns) is a data visualization library built on top of Matplotlib. It provides a high-level interface for drawing attractive statistical graphics.
from sklearn.model_selection import train_test_split #to split data in training set and test set data
#regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#evaluation 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('CCPP_data.csv')

df = pd.DataFrame(data)

#df.info()

print(df.describe())

#print("data frame :\n", df, "\n")


plt.figure(figsize = (7,5)) #This line sets the size of the figure (the entire visualization) to be created. The figsize parameter takes a tuple of width and height in inches.

#df.corr() calculates the correlation matrix of the DataFrame df. The correlation matrix shows how each variable in the DataFrame correlates with every other variable.
#The annot=True parameter adds numerical annotations to the heatmap, displaying the correlation coefficients in each cell.
sns.heatmap(df.corr(), annot=True)



sns.set(style="ticks")
sns.pairplot(df,diag_kind ='hist')
plt.show()

#Model training and testing done in differnt script


