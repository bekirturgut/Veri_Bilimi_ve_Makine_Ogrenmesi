import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

label_encoder = LabelEncoder()
pd.set_option("display.max_columns", None) #bu sayede tüm sutunları görebilcem pandas ayarını değiştirdim

df = pd.read_csv('17-googleplaystore.csv')

# MİSSİNG DATA

print(df.head(),end="\n\n------------------------------------------------\n\n")
print(df.isnull().sum(),end="\n\n------------------------------------------------\n\n")

#df['Reviews'] = df['Reviews'].astype(int) hata aldık demekki hepsi int e çevirmeye uygun değil

print(df["Reviews"].str.isnumeric().sum(),end="\n\n------------------------------------------------\n\n")
#16. satır sonucu 10841 datadan sadece 1i sayıya çevirmeye uygun değil

df_clean = df.copy()
df_clean = df_clean.drop(df_clean.index[10472])
df_clean['Reviews'] = df_clean['Reviews'].astype(int)
print(df_clean.info(),end="\n\n-------------------------------------------------\n\n")

print(df_clean["Size"].value_counts(),end="\n\n-------------------------------------------------\n\n")

df_clean["Size"] = df_clean["Size"].str.replace("M", "000") # bunda M gördüğün yere 000 yaz dedim
df_clean["Size"] = df_clean["Size"].str.replace("k", "") # bunda da k yerine hiçbişi yazma dedim bu sayede int e çevirebilecek hale getirdim
df_clean['Size'] = df_clean['Size'].replace("Varies with device","0") # belirlenmeyenleri 0 yaptım ki int çevirebileyim hepsini
df_clean["Size"] = df_clean["Size"].str.replace(".", "") # . olanları kaldırdım
df_clean['Size'] = df_clean['Size'].astype(int) # boyutları int yaptım
df_clean['Size'] = df_clean['Size'].replace(0,np.nan) # boyut belirlenmeyenleri null yaptım
df_clean["Type"] = label_encoder.fit_transform(df_clean["Type"])

print(df_clean["Size"].unique(),end="\n\n-------------------------------------------------\n\n")

print(df_clean,end="\n\n-------------------------------------------------\n\n")
print(df_clean.info(),end="\n\n-------------------------------------------------\n\n")
print(df_clean["Type"].value_counts(),end="\n\n-------------------------------------------------\n\n")
print(df_clean["Price"].unique(),end="\n\n-----------------------------------------------\n\n")