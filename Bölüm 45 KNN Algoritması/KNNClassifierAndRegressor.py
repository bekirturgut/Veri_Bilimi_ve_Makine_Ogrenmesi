import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,r2_score,mean_absolute_error,mean_squared_error
import builtins
pd.set_option("display.max_columns", None)

def print(*args, **kwargs):
    # Eğer end parametresi belirtilmemişse, otomatik ayarla
    if "end" not in kwargs:
        kwargs["end"] = "\n\n###############################################\n\n"   # Buraya istediğin varsayılanı koy
    return builtins.print(*args, **kwargs)

                # KNN DE BÖYLE KULLANILIYOR

df = pd.read_csv("12-health_risk_classification.csv")

print(df)
print(df.info())
print(df.isnull().sum())
print(df.columns)

X = df.drop("high_risk_flag",axis=1)
y = df["high_risk_flag"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5,algorithm="auto",weights="uniform")
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

df = pd.read_csv("12-house_energy_regression.csv")

print(df)
print(df.info())
print(df.isnull().sum())
print(df.columns)

X = df.drop("daily_energy_consumption_kwh",axis=1)
y = df["daily_energy_consumption_kwh"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
regressor = KNeighborsRegressor(n_neighbors=6,algorithm="auto")
regressor.fit(X_train_scaled, y_train)

y_pred = regressor.predict(X_test_scaled)

print(regressor.score(X_test_scaled, y_pred))
print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
