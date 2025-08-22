import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import builtins
pd.set_option("display.max_columns", None)

def print(*args, **kwargs):
    # Eğer end parametresi belirtilmemişse, otomatik ayarla
    if "end" not in kwargs:
        kwargs["end"] = "\n\n###############################################\n\n"   # Buraya istediğin varsayılanı koy
    return builtins.print(*args, **kwargs)

df = pd.read_csv("11-iris.csv")
df.drop("Id",axis=1,inplace=True)

print(df)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df["Species"].value_counts())

#sns.pairplot(df)
#plt.show()

label_encoder = LabelEncoder()
df["Species"] = label_encoder.fit_transform(df["Species"])

X = df.drop("Species", axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train_scaled, y_train)
y_pred = gnb.predict(X_test_scaled)

from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy Score : ",accuracy_score(y_test,y_pred))