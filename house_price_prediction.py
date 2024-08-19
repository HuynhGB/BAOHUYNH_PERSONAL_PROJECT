import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_df = pd.read_csv("train.csv")
fig, ax = plt.subplots(figsize =(8,5))
sns.heatmap(data_df.isna(), cmap="Blues")
plt.show()

df = data_df.dropna(axis = 1)
print(df.info())

features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
X = data_df[features]
Y = data_df["SalePrice"]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
Y = Y.values
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, train_size = 0.8,random_state = 0)

from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train,Y_train)

Y_pred = dt_model.predict(X_valid)
from sklearn.metrics import mean_squared_error 
print(mean_squared_error(Y_valid,Y_pred))