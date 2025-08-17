import numpy as np
import pandas as pd

weather_df = pd.read_excel('6-weather.xlsx')
print(weather_df,end="\n\n#########################################################\n\n")

employee_df = pd.read_csv('6-employee.csv')
print(employee_df,end="\n\n#########################################################\n\n")


# hatalı yada eksik data

weather_hatali = pd.read_excel('6-weatherna.xlsx')
print(weather_hatali,end="\n\n#########################################################\n\n")
print(weather_hatali.isna(),end="\n\n#########################################################\n\n") # boş eleman olup olmadığını gösterir
print(weather_hatali.describe(),end="\n\n#########################################################\n\n") # sutun analizi verir
print(weather_hatali.dropna(),end="\n\n#########################################################\n\n") # boş elemanı olan astırları siler
print(weather_hatali.fillna(19),end="\n\n#########################################################\n\n") # boş yerlere 19 yaz demek
print(weather_hatali.fillna(weather_hatali.mean()),end="\n\n#########################################################\n\n") # o sutunun ortalamsını koyar

df1 = pd.read_csv('7-concat_data1.csv')
df2 = pd.read_csv('7-concat_data2.csv')

print(df1)
print(df2)

df_concat = pd.concat([df1,df2]) # iki tane dosyanın tablosunu birleştiriyoruz
print(df_concat)
