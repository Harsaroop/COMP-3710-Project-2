import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

raw_data = pd.read_csv('Bias_correction_ucl.csv')
# raw_data.info()
# print(raw_data.columns)
# sns.pairplot(raw_data.sample(2))
x = raw_data[['station', 'Date', 'Present_Tmax', 'Present_Tmin', 'LDAPS_RHmin',
              'LDAPS_RHmax', 'LDAPS_Tmax_lapse', 'LDAPS_Tmin_lapse', 'LDAPS_WS',
              'LDAPS_LH', 'LDAPS_CC1', 'LDAPS_CC2', 'LDAPS_CC3', 'LDAPS_CC4',
              'LDAPS_PPT1', 'LDAPS_PPT2', 'LDAPS_PPT3', 'LDAPS_PPT4', 'lat', 'lon',
              'DEM', 'Solar radiation', 'Next_Tmax', 'Next_Tmin']]
y = raw_data['Slope']
