import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split #to split data in training set and test set data
#regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#evaluation 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from prettytable import PrettyTable


data = pd.read_csv('CCPP_data.csv')
df = pd.DataFrame(data)

#select features.


df_Feat1 = df['AT'].values.reshape(-1,1) ##We select only AT as the predictor (Since it has the strongest correlation with the target variable (PE))
df_Feat2 = df[['AT','V']].values
df_Feat3 = df[['AT','V','RH']].values
df_Feat4 = df[['AT','V','AP', 'RH']].values
predictionY = df['PE'].values

datasets = [df_Feat1, df_Feat2, df_Feat3, df_Feat4]

models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()]

def EvaluateModel(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train) #train
    y_prediction = model.predict(x_test) #predict

    #evaluation
    meanSquareError = mean_squared_error(y_test, y_prediction)
    rootMeanSquareError = np.sqrt(meanSquareError)

    rSquared = r2_score(y_test, y_prediction)
    meanAbsoluteError = mean_absolute_error(y_test, y_prediction)

    #print(f"Model: {type(model).__name__}")
    #print("Means square Error:", rootMeanSquareError)
    #print("R-squared Error:", rSquared)
    #print("Mean absolute error (MAE):", meanAbsoluteError)
    #print("\n")
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]
    table.add_row(["Means square Error", rootMeanSquareError])
    table.add_row(["R-squared Error", rSquared])
    table.add_row(["Mean absolute error (MAE)", meanAbsoluteError])
    print(f"Model: {type(model).__name__}")
    print(table)
    print("\n")
    
              

for ds in datasets:
    x_train, x_test, y_train, y_test = train_test_split(ds, predictionY, test_size = 0.2, random_state =0) #train_test_split(...): Splits the data into random train and test sets. Here, it takes the reshaped df_1 as the input feature (X) and y as the target variable. The test_size=0.2 parameter specifies that 20% of the data will be used for testing, and random_state=0 ensures reproducibility by setting the random seed.
    #X_train, X_test, y_train, y_test: These are the variables that will store the training and testing sets for input features (X) and target variable (y).
    for selectedModel in models:
        EvaluateModel(selectedModel, x_train, x_test, y_train, y_test)
              
    
    
    
    
    


