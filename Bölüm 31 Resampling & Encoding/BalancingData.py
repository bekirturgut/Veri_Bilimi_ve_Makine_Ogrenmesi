import pandas as pd
import numpy as np

#random seed
np.random.seed(42)
set1no = 900
set2no = 100

df1 = pd.DataFrame({
    "feature_1": np.random.normal(loc=0,scale=1,size=set1no), # loc ortlama , scale standart sapma
    "feature_2": np.random.normal(loc=0,scale=1,size=set1no),
    "target": [0] * set1no
})

df2 = pd.DataFrame({
    "feature_1": np.random.normal(loc=0,scale=1,size=set2no), # loc ortlama , scale standart sapma
    "feature_2": np.random.normal(loc=0,scale=1,size=set2no),
    "target": [1] * set2no
})

print(df1.head(),end="\n\n############################\n\n")
print(df2.head(),end="\n\n############################\n\n")

df = pd.concat([df1, df2]).reset_index(drop=True)
print(df,end="\n\n############################\n\n")
print(df["target"].value_counts(),end="\n\n###########################\n\n")

#upsampling --> az olan verileri çoğaltmak
#downsampling --> çok olan verileri azaltmak

df_minority = df[df["target"] == 1]
print(df_minority,end="\n\n############################\n\n")
df_majority = df[df["target"] == 0]
print(df_majority,end="\n\n############################\n\n")

from sklearn.utils import resample

df_minority_upsampled = resample(df_minority,replace=True,n_samples=len(df_majority),random_state=42)
print(df_minority_upsampled,end="\n\n############################\n\n")

df_upsamlped = pd.concat([df_majority,df_minority_upsampled])
print(df_upsamlped,end="\n\n############################\n\n")

print(df_upsamlped["target"].value_counts(),end="\n\n############################\n\n")

#SMOTE (synthetic minority over-sampling technique)
import matplotlib.pyplot as plt

print(df,end="\n\n############################\n\n")
#plt.scatter(df["feature_1"] , df["feature_2"] , c=df["target"])
#plt.show()

from imblearn.over_sampling import SMOTE

oversample = SMOTE()
(x,y) = oversample.fit_resample(df[["feature_1","feature_2"]],df["target"])
print(x,end="\n\n############################\n\n")
print(y,end="\n\n############################\n\n")

print(type(x) , type(y) , end="\n\n############################\n\n")

oversample_df = pd.concat([x,y],axis=1)
print(oversample_df,end="\n\n############################\n\n")
plt.scatter(oversample_df["feature_1"],oversample_df["feature_2"] ,c=oversample_df["target"])
plt.show()