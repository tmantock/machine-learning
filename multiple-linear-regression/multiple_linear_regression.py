#Multiple Linear Regression
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding categorical data
#Encodeing the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
x[:, 3] = labelencoder_X.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
x = onehotencoder.fit_transform(x).toarray()

#Avoiding the Dummy variable Trap
x = x[:, 1:]

#Splitting the dataset into the training and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

#Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""

#Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the Test Set Results
y_pred = regressor.predict(x_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm

#add 1 as a constatnt to the dataset for multiple linear regression
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)
#Create optimal matrix of features
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
#import the the OLS class from statsmodel
#Fit the full model with all possible predictors
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
#Get the summary report from the OLS model and get the p-values
#The significance level will be SL = 0.05 or 5%
regressor_OLS.summary()
#Remove the variable with highest p-value and repeat the process when the hisghest p-value is found for backwards elimination
x_opt = x[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()
#Remove the variable with highest p-value and repeat the process when the hisghest p-value is found for backwards elimination
x_opt = x[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()
#Remove the variable with highest p-value and repeat the process when the hisghest p-value is found for backwards elimination
x_opt = x[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()
#Remove the variable with highest p-value and repeat the process when the hisghest p-value is found for backwards elimination
x_opt = x[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()
#R&D Spend is the most significant independent variable and is a good statistical predictor for profit