import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#df = pd.read_csv("13-WineQT.csv")
#df.hist(figsize=(15,10), bins=30)
#plt.suptitle("WineQT Veri Dağılımları")
#plt.show()

#sns.histplot(x="quality", y="alcohol", data=df)
#plt.title("Alkol % - Kaliteye göre dağılım")
#plt.show()

df = pd.read_csv("13-WineQT.csv")
print(df) # tablo yazdırıyoruz
print(df.describe()) # istatistik bakıyoruz
print(df.isnull().sum()) # null değer sayısına bakıyoruz

plt.figure(figsize = (10,6))
sns.heatmap(df.corr(),annot=True)
plt.show()

print(df.groupby("quality").mean()) # quality e göre diğer özellikleri kıyaslamamıza yarar

sns.pairplot(df) # tüm herşeyin grefiğini çıkartıyor
plt.show()
