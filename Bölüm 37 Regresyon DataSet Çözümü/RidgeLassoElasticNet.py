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
pd.set_option("display.max_rows", None)
df = pd.read_csv("4-Algerian_forest_fires_dataset.csv")
df.loc[:123,"Region"] = 0
df.loc[123:,"Region"] = 1
df.columns = df.columns.str.strip()
df = df.dropna().reset_index(drop=True) # indexleri sıfırladim ve null satırları sildim
df.drop(122,inplace=True)
df[["day","month","year","Temperature","RH","Ws"]] = df[["day","month","year","Temperature","RH","Ws"]].astype(int)
df[["Rain","FFMC","DMC","DC","ISI","BUI","FWI"]] = df[["Rain","FFMC","DMC","DC","ISI","BUI","FWI"]].astype(float)
df["Classes"] = np.where(df["Classes"].str.contains("not fire"),1,0)
df.drop(["day","month","year"],axis=1,inplace=True)
print(df,end="\n\n######################################\n\n")
print(df["Classes"].value_counts(),end="\n\n######################################\n\n")
print(df.info(),end="\n\n######################################\n\n")
print(df.isnull().sum(),end="\n\n######################################\n\n")
"""
sns.heatmap(df.corr()) # renklere göre korelasyon yorumu yapmamızı sağlar ne kadar koyu okadar alakasız demektir
plt.show()
"""
#dependent & independent features

X = df[["Temperature","RH","Ws","Rain","FFMC","DMC","DC","ISI","BUI","Classes","Region"]]
y = df["FWI"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
#print(X_train.corr(),end="\n\n######################################\n\n")

# çok bağımlı olan değişkenleri çıkartıcaz (redundancy,multicollinearity)

# korelasyonu çok önemli bak yüksek korelasyon varsa karşı tarafa sor çıkarabilirsen çıkar

def correlation_for_dropping(df,threshold): # korelasyonu yüksek olanları bu fonksiyon ile buluyoruz
    columns_to_drop = set()
    corr = df.corr()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i,j]) > threshold:
                columns_to_drop.add(corr.columns[i])
    return columns_to_drop
columns_dropping = correlation_for_dropping(X_train,0.85)

X_train.drop(columns_dropping,axis = 1,inplace=True)
X_test.drop(columns_dropping,axis = 1,inplace=True)

print(X_train.head(),end="\n\n######################################\n\n")
print(X_test.head(),end="\n\n######################################\n\n")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""
plt.subplots(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(data=X_train)
plt.title("X_train")
plt.subplot(1,2,2)
sns.boxplot(data=X_train_scaled)
plt.title("X_test")
plt.show()
"""

"""
# LASSO

Lasso = Lasso()
Lasso.fit(X_train_scaled,y_train)   # LASSO
y_pred = Lasso.predict(X_test_scaled)
MAE = mean_absolute_error(y_test,y_pred)
MSE = mean_squared_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print("Mean Absolute Error : ",MAE) # 1.216272633800629
print("Mean Squared Error : ",MSE) # 2.6029377368132196
print("R2 Score : ",score) # 0.9521029422229386
plt.scatter(y_test,y_pred)
plt.show()
"""

"""
# RİDGE

Ridge = Ridge()
Ridge.fit(X_train_scaled,y_train)   # RİDGE
y_pred = Ridge.predict(X_test_scaled)
MAE = mean_absolute_error(y_test,y_pred)
MSE = mean_squared_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print("Mean Absolute Error : ",MAE) #0.7093542448703333
print("Mean Squared Error : ",MSE) #0.8868348464263109
print("R2 Score : ",score) #0.9836812155445575
plt.scatter(y_test,y_pred)
plt.show()
"""

"""
# ELASTİCNET

ElasticNet = ElasticNet()
ElasticNet.fit(X_train_scaled,y_train)  
y_pred = ElasticNet.predict(X_test_scaled)
MAE = mean_absolute_error(y_test,y_pred)
MSE = mean_squared_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print("Mean Absolute Error : ",MAE) #1.749328647109016
print("Mean Squared Error : ",MSE) #5.569160502382824
print("R2 Score : ",score) #0.8975210207375391
"""

"""
# Normal Linear Regression

linear = LinearRegression()
linear.fit(X_train_scaled,y_train)
y_pred = linear.predict(X_test_scaled)
MAE = mean_absolute_error(y_test,y_pred)
MSE = mean_squared_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print("Mean Absolute Error : ",MAE) #0.7036011729575921
print("Mean Squared Error : ",MSE) #0.8764163698605981
print("R2 Score : ",score) #0.9838729275348057
plt.scatter(y_test,y_pred)
plt.show()
"""

"""
# LASSO CROSS VALİDATİON

lassocv = LassoCV(cv=5)
lassocv.fit(X_train_scaled,y_train)
y_pred = lassocv.predict(X_test_scaled)
MAE = mean_absolute_error(y_test,y_pred)
MSE = mean_squared_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
MAE = mean_absolute_error(y_test,y_pred)
MSE = mean_squared_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print("Mean Absolute Error : ",MAE) #0.7222308156305418
print("Mean Squared Error : ",MSE) #0.8972011670293636
print("R2 Score : ",score) #0.9834904632842026
plt.scatter(y_test,y_pred)
plt.show()
"""

"""
# RİDGE CROSS VALİDATİON

ridge = RidgeCV(cv=5)
ridge.fit(X_train_scaled,y_train)
y_pred = ridge.predict(X_test_scaled)
MAE = mean_absolute_error(y_test,y_pred)
MSE = mean_squared_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
MAE = mean_absolute_error(y_test,y_pred)
MSE = mean_squared_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print("Mean Absolute Error : ",MAE) # 0.7093542448703333
print("Mean Squared Error : ",MSE) # 0.8868348464263109
print("R2 Score : ",score) # 0.9836812155445575
plt.scatter(y_test,y_pred)
plt.show()
"""


# ELASTİCNET CROSS VALİDATİON

elastic = ElasticNetCV(cv=5)
elastic.fit(X_train_scaled,y_train)
y_pred = elastic.predict(X_test_scaled)
MAE = mean_absolute_error(y_test,y_pred)
MSE = mean_squared_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
MAE = mean_absolute_error(y_test,y_pred)
MSE = mean_squared_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print("Mean Absolute Error : ",MAE) #0.7354284604918176
print("Mean Squared Error : ",MSE) #0.9249125778630378
print("R2 Score : ",score) #0.9829805413498388
plt.scatter(y_test,y_pred)
plt.show()
