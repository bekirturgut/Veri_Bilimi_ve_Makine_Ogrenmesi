import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
pd.set_option("display.max_columns", None)
df = pd.read_csv("2-multiplegradesdataset.csv")
print(df,end="\n\n")
print(df.describe(),end="\n\n")
print(df.info(),end="\n\n")

#sns.pairplot(df)
#sns.regplot(x = df["Social Media Hours"] , y = df["Exam Score"], )
#plt.show()

# Independent and Dependent Features

X = df.iloc[:,:-1]
y = df.iloc[:,-1] # normal python index ayalarlaması biliyorsun

#X = df[["Study Hours","Sleep Hours","Attendance Rate","Social Media Hours"]]
#y = df["Exam Score"]

# Train - Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.25, random_state = 15)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression

regression = LinearRegression()

regression.fit(X_train,y_train)

new_student = [[5,7,90,5]] # yeni öğrenci tanımladık ama notumu bilmiyoruz
new_student_scaled = scaler.transform(new_student) # verileri transform ettik
print(regression.predict(new_student_scaled),end="\n\n") # eğittiğimiz modele göre verileri verdik bakalım sonuç kaç

# sonuçlar veriyor model çalışıyor ama doğruluk payı hesaplamadık bakalım doğru mu regrasyonlarla bakıcaz bune mse mae rmsr r2adj

# prediction

y_pred = regression.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

MSE = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
R2adj = 1 - ( 1-R2 ) * ( len(y_test) - 1 ) / ( len(y_test) - X_test.shape[1] - 1 ) # Adjusted Score

print("MSE = ",MSE)
print("MAE = ",MAE)
print("R2 = ",R2)
print("R2adj = ",R2adj,end="\n\n")

#plt.scatter(y_test,y_pred)
#plt.show()

"""
residuals = y_test - y_pred
sns.displot(residuals,kind="kde")
plt.show()
"""

print(regression.intercept_) # oluşan fonksiyonun Y yi kestiği noktayı verir
print(regression.coef_) # fonksiyonun x lerinin katsayısını sırasıyla verir

print(df)
