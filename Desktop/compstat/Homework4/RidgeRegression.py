import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
'''
We perform ridge regression on a housing dataset
'''


#Load data
data = pd.read_csv("RealEstate.csv")
# Process data, change categorical variable
short = data[data["Status"] == "ShortSale"]
for local in set(short["Location"]):
    location_list = np.array(short["Location"] == local)
    location_list = location_list.astype(int)
    short.insert(0, local, location_list)

short = short.drop("Location", axis = 1)
short = short.drop("Status", axis = 1)
short.to_csv("ShortSales.csv")
