import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
pd.set_option("display.max_columns", None)

df = pd.read_csv("7-cyber_attack_data.csv")
print(df,end="\n\n##############################################\n\n")

"""
Columns = ['src_packet_rate', 'dst_packet_rate', 'avg_payload_size',
           'connection_duration', 'tcp_flag_count', 'avg_interarrival_time',
           'failed_login_attempts', 'unusual_port_activity_score',
           'session_entropy', 'avg_response_delay', 'attack_type']
           
src_packet_rate -> Kaynak tarafındaki paket gönderim hızı
dst_packet_rate -> Hedef tarafındaki paket alma hızı
avg_payload_size -> Paketlerdeki ortalama yük (payload) boyutu
connection_duration -> Bağlantının süresi (saniye cinsinden)
tcp_flag_count -> TCP bayraklarının (flag) görülme sayısı
avg_interarrival_time -> Paketler arası geliş süresi
failed_login_attempts -> Başarısız giriş denemelerinin sayısı
unusual_port_activity_score -> Olağandışı port kullanımı puanı
session_entropy -> Oturum davranışının entropisi (anormallik tespiti için)
avg_response_delay -> Sunucu yanıtındaki ortalama gecikme (ms cinsinden)
attack_type -> 0 = Normal, 1 = DDoS, 2 = Port Tarama
"""

########################################################################################################################

X = df.drop("attack_type",axis=1)
y = df["attack_type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

score = accuracy_score(y_pred,y_test)
print(score,end="\n\n##############################################\n\n")
print(classification_report(y_pred,y_test),end="\n\n##############################################\n\n")
print(confusion_matrix(y_pred,y_test),end="\n\n##############################################\n\n")

########################################################################################################################

# HYPERPARAMETER TUNİNG

"""
LogisticRegression() 'un solver parametresi 

    solver           penalty          multiniomal multiclass
   --------         ---------        ------------------------
    'lbfgs'         'l2',none                  yes 
   'liblinear'      'l1','l2'                  no
   'newton-cg'      'l2',none                  yes
'newton-cholesky'   'l2',none                  no
     'sag'          'l2',none                  yes
     'saga' 'elasticnet','l1','l2',none        yes

"""

model_hyperparamater_tuning = LogisticRegression()

penalty = ["l1","l2","elasticnet"] # ceza verme şekilleri
c_values = [100,10,1,0.1,0.01] # ceza verme kaysayıları
solver = ['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga']

"""
bu üstteki 3 listeyi tanımladık ama her kombinasyon farklı bir değer çıkartıcak verimi için 
tüm kombinasyonlara bakmamız lazım bunun için grid-search diye bir kütüphane var ve onu kullanıcaz
bu kütüphane de bizden sözlük halinde istiyor onun içinde sözlüğe çeviricez bu 3 listeyi
"""

params = dict(penalty=penalty,C=c_values,solver=solver)

########################################################################################################################

# GRİD SEARCH CV

from sklearn.model_selection import GridSearchCV,StratifiedKFold

cv = StratifiedKFold()
grid = GridSearchCV(model_hyperparamater_tuning,param_grid=params,cv=cv,scoring="accuracy",n_jobs=-1)

"""
scoring = hangi hesaplamaya göre en iyisini bulucak anlamında
n_jobs = -1 yapınca tüm CPU yu kullanıcak yaparken anlamında 
"""

grid.fit(X_train,y_train)

print(grid.best_params_,end="\n\n##############################################\n\n")
print(grid.best_score_,end="\n\n##############################################\n\n")

"""
grid.bst_params_ = en iyi kombinasyonu verir
çıktı = {'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear'}
----------------------------------------------------------
grid.best_score_ = en iyi score değerini verir 
çıktı = 0.9242857142857142
"""

y_pred_grid = grid.predict(X_test)

########################################################################################################################

score = accuracy_score(y_pred_grid,y_test)
print(score,end="\n\n##############################################\n\n")
print(classification_report(y_pred_grid,y_test),end="\n\n##############################################\n\n")
print(confusion_matrix(y_pred_grid,y_test),end="\n\n##############################################\n\n")

########################################################################################################################

from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier

# ONE vs REST && ONE vs ONE

onevsone_model = OneVsOneClassifier(LogisticRegression())

onevsrest_model = OneVsOneClassifier(LogisticRegression())

onevsone_model.fit(X_train,y_train)
y_pred_onevsone = onevsone_model.predict(X_test)
score = accuracy_score(y_pred_grid,y_test)
print(score,end="\n\n##############################################\n\n")
print(classification_report(y_pred_grid,y_test),end="\n\n##############################################\n\n")
print(confusion_matrix(y_pred_grid,y_test),end="\n\n##############################################\n\n")

onevsrest_model.fit(X_train,y_train)
y_pred_onevsrest = onevsrest_model.predict(X_test)
score = accuracy_score(y_pred_grid,y_test)
print(score,end="\n\n##############################################\n\n")
print(classification_report(y_pred_grid,y_test),end="\n\n##############################################\n\n")
print(confusion_matrix(y_pred_grid,y_test),end="\n\n##############################################\n\n")