import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.stats import norm
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression

# Read the csv file
data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

#Exploratory Data Analysis
data.info()
data.head()
data.tail()
data.describe().T
data.isnull().sum()

data['Sleep Disorder'] = data['Sleep Disorder'].fillna('No Disorder')
data.isnull().sum()
data.head()

data.drop('Person ID',axis=1,inplace=True)
data.columns = ['gender', 'age', 'occupation', 'sleep_duration', 'sleep_quality', 'physical_activity_level','stress_level', 'bmi_category', 'blood_pressure', 'heart_rate', 'daily_steps', 'sleep_disorder']

#Table
data.head()

#Figure
corr = data.select_dtypes(include=['number']).corr(method='pearson')
mask = np.tril(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(15, 12))
sns.heatmap(corr, mask=mask, vmax=0.9, square=True, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.show()