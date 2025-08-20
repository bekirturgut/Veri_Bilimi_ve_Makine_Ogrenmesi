import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.pipeline import Pipeline

pd.set_option("display.max_columns", None)

df = pd.read_csv("3-customersatisfaction.csv")
df.drop("Unnamed: 0",axis=1,inplace=True)
print(df,end="\n\n########################################################\n\n")

"""
plt.scatter(df["Customer Satisfaction"],df["Incentive"])
plt.xlabel("Customer Satisfaction")
plt.ylabel("Incentive")
plt.show()
"""

X = df[["Customer Satisfaction"]]
y = df["Incentive"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

regression = LinearRegression()

regression.fit(X_train,y_train)

y_pred = regression.predict(X_test)

MSE = mean_squared_error(y_test,y_pred)
MAE = mean_absolute_error(y_test,y_pred)
R2 = r2_score(y_test,y_pred)
R2adj = 1 - ( 1-R2 ) * ( len(y_test) - 1 ) / ( len(y_test) - X_test.shape[1] - 1 )

print("MSE = ",MSE)
print("MAE = ",MAE)
print("R2 = ",R2)
print("R2adj = ",R2adj,end="\n\n########################################################\n\n")
"""
plt.scatter(X_train,y_train)
plt.plot(X_train,regression.predict(X_train))
plt.show()
"""
print("regression.intercept_ = ",regression.intercept_)
print("regression.coef_ = ",regression.coef_,end="\n\n########################################################\n\n")

poly = PolynomialFeatures(degree=2,include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

regression.fit(X_train_poly,y_train)

y_pred_poly = regression.predict(X_test_poly)

print(X_train_poly,end="\n\n########################################################\n\n")

MSE = mean_squared_error(y_test,y_pred_poly)
MAE = mean_absolute_error(y_test,y_pred_poly)
R2 = r2_score(y_test,y_pred_poly)
R2adj = 1 - ( 1-R2 ) * ( len(y_test) - 1 ) / ( len(y_test) - X_test.shape[1] - 1 )

print("MSE = ",MSE)
print("MAE = ",MAE)
print("R2 = ",R2)
print("R2adj = ",R2adj,end="\n\n########################################################\n\n")

print("regression.intercept_ = ",regression.intercept_)
print("regression.coef_ = ",regression.coef_,end="\n\n########################################################\n\n")

"""
plt.scatter(X_train,y_train)
plt.scatter(X_train,regression.predict(X_train_poly) , color = "r")
plt.show()
"""

new_df = pd.read_csv("3-newdatas.csv")
new_df.rename(columns = {"0" : "Customer Satisfaction"} , inplace = True)

X_new = new_df[["Customer Satisfaction"]]
X_new = scaler.fit_transform(X_new)
X_new_poly = poly.transform(X_new)
y_new = regression.predict(X_new_poly)

"""
plt.plot(X_new,y_new,"r" , label = "New Predictions")
plt.scatter(X_train,y_train, label = "Training Points")
plt.show()
"""

# PİPLİNE

def poly_regression(degree):
    poly_features = PolynomialFeatures(degree)
    lin_reg = LinearRegression()
    scaler = StandardScaler()
    pipline = Pipeline(
        [
            ("standart_scaler", scaler),
            ("poly_features", poly_features),
            ("lin_reg", lin_reg)
        ]
    )
    pipline.fit(X_train,y_train)
    score = pipline.score(X_test,y_test)

    y_pred_new = pipline.predict(X_new)
    plt.plot(X_new, y_pred_new, "r", label="New Predictions")
    plt.scatter(X_train, y_train, label="Training Points")
    plt.scatter(X_test,y_test, color = "r")
    plt.legend()
    plt.show()
    return score


print("Degree 0 = ",poly_regression(0))
print("Degree 1 = ",poly_regression(1))
print("Degree 2 = ",poly_regression(2))
print("Degree 3 = ",poly_regression(3))
print("Degree 4 = ",poly_regression(4))
print("Degree 5 = ",poly_regression(5))
print("Degree 6 = ",poly_regression(6))
print("Degree 7 = ",poly_regression(7))
print("Degree 8 = ",poly_regression(8))
print("Degree 9 = ",poly_regression(9))
print("Degree 10 = ",poly_regression(10))





