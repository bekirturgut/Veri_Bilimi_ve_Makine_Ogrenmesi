import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.linear_model import Lasso,LassoCV
from sklearn.linear_model import ElasticNet,ElasticNetCV
pd.set_option("display.max_columns", None)


df = pd.read_csv("weatherHistory.csv")

map = {
    "Partly Cloudy":"1",
    "Mostly Cloudy":"2",
    "Overcast":"3",
    "Clear":"4",
    "Foggy":"5",
    "Breezy and Overcast":"6",
    "Breezy and Mostly Cloudy":"7",
    "Breezy and Partly Cloudy":"8",
    "Dry and Partly Cloudy":"9",
    "Windy and Partly Cloudy":"10",
    "Light Rain":"11",
    "Breezy":"12",
    "Windy and Overcast":"13",
    "Humid and Mostly Cloudy":"14",
    "Drizzle":"15",
    "Breezy and Foggy":"16",
    "Windy and Mostly Cloudy":"17",
    "Dry":"18",
    "Humid and Partly Cloudy":"19",
    "Dry and Mostly Cloudy":"20",
    "Rain":"21",
    "Windy":"22",
    "Humid and Overcast":"23",
    "Windy and Foggy":"24",
    "Dangerously Windy and Partly Cloudy":"25",
    "Windy and Dry":"26",
    "Breezy and Dry":"27",

}

print(df.head())
print(df.columns)

df.drop(["Formatted Date","Loud Cover"] , axis = 1, inplace = True)
df.dropna(inplace = True)
df["Summary"] = df["Summary"].replace(map).astype(int)
df["Precip Type"] = np.where(df["Precip Type"].str.contains("rain"),1,0)
df.drop("Daily Summary",axis=1,inplace = True)


print(df,end="\n\n#############################################################\n\n")
print(df.info(),end="\n\n#############################################################\n\n")
print(df.shape,end="\n\n#############################################################\n\n")

X = df[['Summary', 'Precip Type', 'Temperature (C)',
       'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)',
       'Visibility (km)', 'Pressure (millibars)']]
y = df['Apparent Temperature (C)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

linear = LinearRegression()
linear.fit(X_train,y_train)
y_pred = linear.predict(X_test)
MAE = mean_absolute_error(y_test,y_pred)
MSE = mean_squared_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print("Mean Absolute Error : ",MAE,end="\n\n#############################################################\n\n")
print("Mean Squared Error : ",MSE,end="\n\n#############################################################\n\n")
print("R2 Score : ",score,end="\n\n#############################################################\n\n")

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label="Tahminler")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=5, label="İdeal")
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.title("Gerçek vs Tahmin Edilen Apparent Temperature")
plt.legend()
plt.show()
