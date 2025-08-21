import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
pd.set_option("display.max_columns", None)

df = pd.read_csv("6-bank_customers.csv")

print(df,end="\n\n##############################################\n\n")
print(df.info(),end="\n\n##############################################\n\n")
print(df.isnull().sum(),end="\n\n##############################################\n\n")

X = df.drop("subscribed",axis=1)
y = df["subscribed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""
X_train = çalışacağı veriseti
X_test = deneyeceği veriseti
y_train = çalışacağı veriseti
y_test = deneyeceği veriseti
"""

logistic = LogisticRegression()
logistic.fit(X_train,y_train)
y_pred = logistic.predict(X_test)

score = accuracy_score(y_pred,y_test) # 0.9166666666666666 ==> %92 başarı oranı

print(score,end="\n\n##############################################\n\n")
print(classification_report(y_pred,y_test),end="\n\n##############################################\n\n")
print(confusion_matrix(y_pred,y_test),end="\n\n##############################################\n\n")

"""
confusion_matrix ==> 146 = 1,1 | 14 = 1,0 | 11 = 0,1 | 129 = 0,0

yani;
-146 sına olcak demiş olmuş
-14 üne olcak demiş olmamış 
-11 ine olmiycak demiş olmuş 
-129 una olmayacak demiş olmamış 

yeni bize lazım olan kısım 1,1 ve 0,0 olan kısım ve oranı yüksek yani başarılı

[[146  14]
 [ 11 129]]
"""

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

y_pred = grid.predict(X_test)

########################################################################################################################

# RANDOM SEARCH CV

from sklearn.model_selection import RandomizedSearchCV

model_random_search_cv = LogisticRegression()
random_cv = RandomizedSearchCV(model_random_search_cv,params , cv=5,n_iter=10, scoring="accuracy")
random_cv.fit(X_train,y_train)

print(random_cv.best_params_,end="\n\n##############################################\n\n") # çıktı = {'solver': 'saga', 'penalty': 'l1', 'C': 0.1}
print(random_cv.best_score_,end="\n\n##############################################\n\n") # çıktı = 0.9214285714285714


