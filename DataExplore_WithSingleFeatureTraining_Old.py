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

df_1 = df['AT'] #We select only AT as the predictor (Since it has the strongest correlation with the target variable (PE))
df_2 = df[['AT','V']]
df_3 = df[['AT','V','RH']] #We select AT, V and RH as the predictors
df_4 = df[['AT', 'V', 'AP', 'RH']] #We select AT, V, AP, and RH as the predictors
y = df['PE'] #Our target variable (PE) is y

X_train, X_test, y_train, y_test = train_test_split(df_1.values.reshape(-1, 1),y,test_size = 0.2, random_state =0)
#This step will be repeated for all the other models’ datasets by replacing df_1 with the corresponding features array. For instance, Model 4 with df_4.

#For Linear Regression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
#Now that we have trained our model on our training set, we are ready to make predictions on our test set (our model has never seen this set before)

y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred) #Root Mean Squared Error (RMSE): measures the average error performed by the model in predicting the outcome for an observation.
rmse = np.sqrt(mse)

r_squared = r2_score(y_test,y_pred) #It means how much of the variation in the target variable that can be explained by the set of features used in training the model.

mae= mean_absolute_error(y_test, y_pred)

print("Regression")
#print("Regression results for df1", y_pred)
print("Means square Error df1", rmse)
print("R-squared Error df1", r_squared)
print("Mean absoute error (MAE) df1", mae)

print("")
#for Decision Tree Regression
dt_regressor = DecisionTreeRegressor()
dt_regressor.fit(X_train, y_train)

y_pred = dt_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(y_test,y_pred)

mae= mean_absolute_error(y_test, y_pred)
print("Decision Tree")
#print("Decision Tree results for df1", y_pred)
print("Decision Tree Means square Error df1", rmse)
print("R-squared Error df1", r_squared)
print("Mean absoute error (MAE) df1", mae)

print("")
#For Randome Forest Regression
rf_regressor = RandomForestRegressor()
rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(y_test,y_pred)

mae= mean_absolute_error(y_test, y_pred)
print("Random Forest")
#print("Random Forest results for df1", y_pred)
print("Means square Error df1", rmse)
print("R-squared Error df1", r_squared)
print("Mean absoute error (MAE) df1", mae)


