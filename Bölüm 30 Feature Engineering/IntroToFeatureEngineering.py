import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

df = sns.load_dataset("titanic")
print(df.isnull().sum()) # null değer sayıları verilir

df["age_mean"] = df["age"].fillna(df["age"].mean())
df["age"].fillna(df["age"].mean() , inplace = True)
sns.histplot(data=df["age"],kde=True )
print(df[["age","age_mean"]])
plt.show()