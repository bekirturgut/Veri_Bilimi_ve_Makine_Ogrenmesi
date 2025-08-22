import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
import builtins
pd.set_option("display.max_columns", None)

def print(*args, **kwargs):
    # Eğer end parametresi belirtilmemişse, otomatik ayarla
    if "end" not in kwargs:
        kwargs["end"] = "\n\n###############################################\n\n"   # Buraya istediğin varsayılanı koy
    return builtins.print(*args, **kwargs)


df = pd.read_csv("10-diamonds.csv")
df.drop("Unnamed: 0",axis=1,inplace=True)
df = df.drop(df[df["x"] ==0].index)
df = df.drop(df[df["y"] ==0].index)
df = df.drop(df[df["z"] ==0].index)
df = df[(df["depth"] < 75) & (df["depth"] > 45)]
df = df[(df["table"] < 75) & (df["table"] > 40)]
df = df[(df["z"] < 30) & (df["z"] > 2)]
df = df[(df["y"] < 20)]

print(df)
print(df.shape)
print(df.columns)
print(df.info())

#sns.pairplot(df)
#plt.show()

X = df.drop("price",axis=1)
Y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=15)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for col in ["cut","color","clarity"]:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


################################################
"""
                # LİNEAR DENEME = İYİ %86

from sklearn.linear_model import LinearRegression

linear = LinearRegression()
linear.fit(X_train_scaled, y_train)
y_pred = linear.predict(x_test_scaled)
MAE = mean_absolute_error(y_pred, y_test)
MSE = mean_squared_error(y_pred, y_test)
R2 = r2_score(y_pred, y_test)
print("MAE = ",MAE)
print("MSE = ",MSE)
print("R2 = ",R2)

plt.scatter(y_test,y_pred)
plt.show()
"""
################################################
"""
                    # SVR DENEMESİ = BOK
        
from sklearn.svm import SVR

svr = SVR()
svr.fit(X_train_scaled, y_train)
y_pred_svr = svr.predict(x_test_scaled)
MAE = mean_absolute_error(y_test, y_pred_svr)
MSE = mean_squared_error(y_test, y_pred_svr)
R2 = r2_score(y_test, y_pred_svr)
print("MAE = ",MAE)
print("MSE = ",MSE)
print("R2 = ",R2)
plt.scatter(y_test,y_pred_svr)
plt.show()
"""
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

param_grid = {
    "C" : [0.1,1,10,100,1000],
    "gamma" : [1,0.1,0.001],
    "kernel" : ["linear","rbf"]
}

grid = GridSearchCV(estimator=SVR(), param_grid=param_grid,n_jobs=-1,verbose=3)
grid.fit(X_train_scaled, y_train)
print(grid.best_params_)
y_pred_grid = grid.predict(x_test_scaled)
MAE = mean_absolute_error(y_test, y_pred_grid)
MSE = mean_squared_error(y_test, y_pred_grid)
R2 = r2_score(y_test, y_pred_grid)
print("MAE = ",MAE)
print("MSE = ",MSE)
print("R2 = ",R2)
plt.scatter(y_test,y_pred_grid)
plt.show()