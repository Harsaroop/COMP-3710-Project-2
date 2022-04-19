import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

raw_data = pd.read_csv('Bias_correction_ucl.csv')
#raw_data.info()
sns.pairplot(raw_data)