from re import T
from matplotlib.colors import Normalize
from matplotlib.patheffects import Normal
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Load the data
data = pd.read_csv('some_data.csv',index_col=0)
data = data.T
print(data.head())

data.fillna(method='ffill',inplace=True)


years = np.array(data.index)
years = [int(i) for i in years]
years = np.array(years)
CDR = np.array(data["CDR"])
male_death = np.array(data["male_mortality"])
female_death = np.array(data["female_mortality"])

# smooth out the data

kernal = np.ones(5)/5

CDR_smooth = np.convolve(CDR,kernal,'same')
male_smooth = np.convolve(male_death,kernal,'same')
female_smooth = np.convolve(female_death,kernal,'same')
CDR_smooth = CDR_smooth/np.linalg.norm(CDR_smooth)
male_smooth = male_smooth/np.linalg.norm(male_smooth)
female_smooth = female_smooth/np.linalg.norm(female_smooth)

# Plot the data
plt.plot(CDR_smooth,label='CDR')
plt.plot(male_smooth,label='male_mortality')
plt.plot(female_smooth,label='female_mortality')
plt.grid(visible=True)
plt.legend()
plt.show()