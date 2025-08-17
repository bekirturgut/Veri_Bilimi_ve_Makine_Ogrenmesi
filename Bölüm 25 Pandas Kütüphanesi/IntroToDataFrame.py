import numpy as np
import pandas as pd

data = np.random.randint(0,10000,size=(4,4))
print(data,end="\n\n####################################\n\n")

data_frame = pd.DataFrame(data)  # series tek boyutlu dataları tutar ama 2 boyutlu ise dataframe tutar
print(data_frame,end="\n\n####################################\n\n")
print(data_frame[0],end="\n\n####################################\n\n")

data_frame1 = pd.DataFrame(data,index=["bekir","turgut","sude","emine"],columns=["Salary","Age","AgeYear","Grade"])
print(data_frame1,end="\n\n#####################################\n\n")
print(data_frame1.iloc[0],end="\n\n#####################################\n\n")  # 0. indexteki tüm bilgileri getiriyor mukemmel bişi
print(data_frame1["Salary"],end="\n\n#####################################\n\n")

data_frame1["tutar"] = 10  # tutar adında bir sutun ekleyerek tüm satırlarına 10 yazdı
print(data_frame1,end="\n\n#####################################\n\n")

data_frame1.drop("tutar",axis=1,inplace=True) # tutarı sil , axis 0 = satır , axis 1 = sutun , inplace son halinde yeni ürün oluşturma üstüne kaydet demek
print(data_frame1,end="\n\n#####################################\n\n")

data_frame1.drop("emine",axis=0,inplace=True)
print(data_frame1,end="\n\n#####################################\n\n")  # emine satırını sildi
print(data_frame1.loc["bekir"]["Salary"],end="\n\n#####################################\n\n")

data_frame1.loc["bekir","Salary"] = 2003 # bekirin salary kısmını 2003 yaptık
print(data_frame1,end="\n\n#####################################\n\n")
