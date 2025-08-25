import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import builtins
pd.set_option("display.max_columns", None)

def print(*args, **kwargs):
    # Eğer end parametresi belirtilmemişse, otomatik ayarla
    if "end" not in kwargs:
        kwargs["end"] = "\n\n###############################################\n\n"   # Buraya istediğin varsayılanı koy
    return builtins.print(*args, **kwargs)

df = pd.read_csv("15-gym_crowdedness.csv")
df["date"] = pd.to_datetime(df["date"],utc=True)
df["year"] = df["date"].dt.year
df.drop("date",axis=1,inplace=True)
df.drop("timestamp",axis=1,inplace=True)

print(df)
print(df.info())
print(df.isnull().sum())
print(df.columns)

X = df.drop("number_people",axis=1)
y = df["number_people"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

def calculate_model_metrics(true,predict):
    mae = mean_absolute_error(true,predict)
    mse = mean_squared_error(true,predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(true,predict)
    return mae,mse,rmse,r2

models = {
    "Linear Regression" : LinearRegression(),
    "Ridge" : Ridge(),
    "Lasso" : Lasso(),
    "Decision Tree" : DecisionTreeRegressor(),
    "Random Forest Regressor" : RandomForestRegressor(),
    "K-Neighbors Regressor" : KNeighborsRegressor()
}

for i in range(len(models)):
    model = list(models.values())[i]
    model.fit(X_train,y_train)
    y_train_pred = model.predict(X_train) # overfittingi daha rahat kontorl etmek için bunu yaptık
    y_test_pred = model.predict(X_test)

    mae, mse, rmse, r2 = calculate_model_metrics(y_train,y_train_pred)
    print("MAE = ",mae)
    print("MSE = ",mse)
    print("RMSE = ",rmse)
    print("R2 = ",r2)
    mae, mse, rmse, r2 = calculate_model_metrics(y_test,y_test_pred)
    print("MAE = ",mae)
    print("MSE = ",mse)
    print("RMSE = ",rmse)
    print("R2 = ",r2)






