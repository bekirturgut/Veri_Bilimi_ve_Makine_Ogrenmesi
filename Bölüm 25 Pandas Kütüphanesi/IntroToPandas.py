import pandas as pd
import numpy as np

grades = {"bekir":50,"sude":100,"turgut":90}

print(pd.Series(grades),end="\n\n#######################\n\n") # bir seri oluşturmaya yarıyor

names = ["bekir","sude","turgut"]
grades1 = [50,100,90]

print(pd.Series(names),end="\n#######################\n\n")
print(pd.Series(grades1),end="\n########################\n\n")
print(pd.Series(names,grades1),end="\n########################\n\n")

nmpy_array = np.array([1,2,5,6])
print(pd.Series(nmpy_array),end="\n########################\n\n")

result = pd.Series(data=[10,5,100],index = ["bekir","sude","emine"])
result2 = pd.Series(data=[20,50,10],index = ["bekir","sude","emine"])
result3 = result + result2
print(result3)