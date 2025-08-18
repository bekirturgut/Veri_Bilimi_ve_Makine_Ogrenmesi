import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

df = sns.load_dataset("titanic")
print(df)

print(df.isnull().sum())
df[["embark_town"]].dropna()
df_onehot = pd.get_dummies(df , columns=["sex","embark_town"])

#son satırın olayı şu male ve female olarak string tutulmuş çok yer kaplıyor bu kodda sex_male olarak
#bir sutun açıyor ve male olanlara true olmayanlara false koyuyor bunun sonucunda yerden
#tasarruf yapmış oluruz

print(df_onehot)

label_encoder = LabelEncoder()
df_label = df.copy()
df_label["sex"] = label_encoder.fit_transform(df_label["sex"]) # istenilen kısımdaki yerleri diziye çevirir

#son satırın olayı şu male ve female olarak string tutulmuş çok yer kaplıyor bu kodda
#sex kısmına male olanlara 1 olmayanlara 0 koyuyor bunun sonucunda yerden baya
#tasarruf yapmış oluruz

print(df_label)