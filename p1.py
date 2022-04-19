import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

raw_data = pd.read_csv('Bias_correction_ucl.csv')
# raw_data.info()
# print(raw_data.columns)
# sns.pairplot(raw_data.sample(2))

# Building a Machine Learning Linear Regression Model
x = raw_data[['station', 'Date', 'Present_Tmax', 'Present_Tmin', 'LDAPS_RHmin',
              'LDAPS_RHmax', 'LDAPS_Tmax_lapse', 'LDAPS_Tmin_lapse', 'LDAPS_WS',
              'LDAPS_LH', 'LDAPS_CC1', 'LDAPS_CC2', 'LDAPS_CC3', 'LDAPS_CC4',
              'LDAPS_PPT1', 'LDAPS_PPT2', 'LDAPS_PPT3', 'LDAPS_PPT4', 'lat', 'lon',
              'DEM', 'Solar radiation', 'Next_Tmax', 'Next_Tmin']]
y = raw_data['Slope']

# Splitting our Data Set into Training Data and Test Data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Building and Training the Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
print(model.coef_)

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

# from sklearn.linear_model import LinearRegression

# model = LinearRegression()
# model.fit(x_train, y_train)
# print(model.coef_)
# print(model.intercept_)

# pd.DataFrame(model.coef_, x.columns, columns = ['Coeff'])

# predictions = model.predict(x_test)

# # plt.scatter(y_test, predictions)
# plt.hist(y_test - predictions)

# from sklearn import metrics

# metrics.mean_absolute_error(y_test, predictions)

# metrics.mean_squared_error(y_test, predictions)

# np.sqrt(metrics.mean_squared_error(y_test, predictions))