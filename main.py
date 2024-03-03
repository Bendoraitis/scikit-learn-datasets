from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load the California Housing dataset
california_housing = fetch_california_housing()
data = california_housing.data
feature_names = california_housing.feature_names
'''
:Attribute Information:
    - MedInc        median income in block group
    - HouseAge      median house age in block group
    - AveRooms      average number of rooms per household
    - AveBedrms     average number of bedrooms per household
    - Population    block group population
    - AveOccup      average number of household members
    - Latitude      block group latitude
    - Longitude     block group longitude
'''

df_california = pd.DataFrame(data, columns=feature_names)

print(feature_names)
print(df_california)
