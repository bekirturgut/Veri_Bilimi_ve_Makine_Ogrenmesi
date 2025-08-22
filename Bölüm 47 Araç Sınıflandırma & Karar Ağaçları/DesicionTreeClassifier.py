import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

pd.set_option("display.max_columns", None)
import builtins
def print(*args, **kwargs):
    # Eğer end parametresi belirtilmemişse, otomatik ayarla
    if "end" not in kwargs:
        kwargs["end"] = "\n\n###############################################\n\n"   # Buraya istediğin varsayılanı koy
    return builtins.print(*args, **kwargs)


df = pd.read_csv("13-car_evaluation.csv")

df["2"] = df["2"].replace("5more","5").astype(int)
df["2.1"] = df["2.1"].replace("more","5").astype(int)

print(df)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.columns)

for col in df.columns:
    print(col,end="")
    print(df[col].unique())

X = df.drop("unacc",axis=1)
y = df["unacc"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

# ORDİNAL ENCODİNG

categorical_cols = ['vhigh' , 'vhigh.1' , 'small' , 'low']
numerical_cols = ["2","2.1"]

ordinal_encoder = OrdinalEncoder(
    categories= [
        ['vhigh' 'high' 'med' 'low'] ,
        ['vhigh' 'high' 'med' 'low'] ,
        ['small' 'med' 'big'] ,
        ['med' 'high' 'low']
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("transformation_name_not_matter",ordinal_encoder,categorical_cols),
    ],
    remainder="passthrough"
)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(criterion="gini",max_depth=3,random_state=0)
tree_model.fit(X_train_transformed, y_train)
y_pred = tree_model.predict(X_test_transformed)

print(tree_model.score(X_test_transformed, y_test))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

