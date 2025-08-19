import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
pd.set_option("display.max_columns", None)

df = pd.read_csv("1-studyhours.csv")
#print(df)

plt.scatter(df["Study Hours"],df["Exam Score"])
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.grid(True)
plt.show()

# Independent and Dependent Features

X = df[["Study Hours"]] # Raconda X dataframe olması lazım
y = df["Exam Score"]

print(type(X), type(y) , end="\n\n")

# Test - Train Split
# bizim bir test için birde eğitmek için iki parça lazım bunun içinde alttaki importu yapmamız lazım

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# Standardize the data set

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train,end="\n\n")
print(X_test,end="\n\n")

# Gelelim Modeli Eğitmeye OLEEEYYYY

from sklearn.linear_model import LinearRegression

regression = LinearRegression()

regression.fit(X_train,y_train)
#print("Coefficients: ", regression.coef_)
#print("Intercept: ", regression.intercept_,end="\n\n")
# Bu ne Anlama Geliyor kosun sonucunda X = 16.18 , Y = 76.91 , Denklem = 76.91 + 16.18x

#plt.scatter(X_train,y_train)
#plt.plot(X_train,regression.predict(X_train),color="red")
#plt.show()

# X = 20 , Y = ?

#print(regression.predict(scaler.transform([[20]])))  # 20 saat çalışan birinin tahmini alacağı not

y_pred_test = regression.predict(X_test)

plt.scatter(y_pred_test,y_test)
plt.show()

from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score

mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
print("MSE = ",mse)
print("MAE = ",mae)
print("RMSE = ",rmse)

r2 = r2_score(y_test, y_pred_test)
print("R2 = ",r2)
print("R2adj = ",1 - ( 1-r2 ) * ( len(y_test) - 1 ) / ( len(y_test) - X_test.shape[1] - 1 ))

# 1 - ( 1-model.score(X, y) ) * ( len(y) - 1 ) / ( len(y) - X.shape[1] - 1 )
# bu adjusted r score formülünün python hali internetten bulunuyor çünkü kütüphanede yok


