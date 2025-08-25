from cmath import nan

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from commctrl import TTF_TRACK
from narwhals.selectors import categorical

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import builtins
pd.set_option("display.max_columns", None)

def print(*args, **kwargs):
    # Eğer end parametresi belirtilmemişse, otomatik ayarla
    if "end" not in kwargs:
        kwargs["end"] = "\n\n###############################################\n\n"   # Buraya istediğin varsayılanı koy
    return builtins.print(*args, **kwargs)

########################################################################################################################
col_names = ["age","workclass","finalweight","education","education_num","marital_status","occupation",
             "relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","income"]
df = pd.read_csv("14-income_evaluation.csv")
df.columns = col_names

print(df)
print(df.info())
print(df.columns)

categorical = [col for col in df.columns if df[col].dtype == "O"]
numerical = [col for col in df.columns if df[col].dtype != "O"]

df["workclass"] = df["workclass"].replace(" ?", np.nan)
df["occupation"] = df["occupation"].replace(" ?", np.nan)
df["native_country"] = df["native_country"].replace(" ?", np.nan)

##################################################################################

X = df.drop("income",axis=1)
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

categorical = [col for col in X_train.columns if X_train[col].dtype == "O"]

for i in [X_train,X_test]:
    i["workclass"] = i["workclass"].fillna(X_train["workclass"].mode()[0])
    i["occupation"] = i["occupation"].fillna(X_train["occupation"].mode()[0])
    i["native_country"] = i["native_country"].fillna(X_train["native_country"].mode()[0])


# encoding

y_train_binary = y_train.apply(lambda x: 1 if x.strip() == ">50k" else 0)
target_means = y_train_binary.groupby(X_train["native_country"]).mean()
X_train["native_country_encoded"] = X_train["native_country"].map(target_means)
X_train["native_country_encoded"] = X_train["native_country_encoded"].fillna(y_train_binary.mean())
X_test["native_country_encoded"] = X_train["native_country"].map(target_means)
X_test["native_country_encoded"] = X_train["native_country_encoded"].fillna(y_train_binary.mean())

X_train.drop("native_country",axis=1,inplace=True)
X_test.drop("native_country",axis=1,inplace=True)

print(X_train.info())
print(X_test.info())

one_hot_categories = ['workclass','education','marital_status','occupation', 'relationship', 'race', 'sex']

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

encoder = ColumnTransformer(
    transformers=[
        ("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),one_hot_categories)
    ],remainder="passthrough"
)

X_train_enc = encoder.fit_transform(X_train)
X_test_enc = encoder.transform(X_test)

columns = encoder.get_feature_names_out()
X_train = pd.DataFrame(X_train_enc, columns=columns, index=X_train.index)
X_test = pd.DataFrame(X_test_enc, columns=columns, index=X_test.index)

cols = X_train.columns

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=cols)
X_test = pd.DataFrame(X_test, columns=cols)

# Training

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=15,n_estimators=100)

rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy Score : ",accuracy_score(y_test,y_pred))
