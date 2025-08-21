import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from stack_data import markers_from_ranges

pd.set_option("display.max_columns", None)

df = pd.read_csv("8-fraud_detection.csv")

print(df,end="\n\n#####################################################################################\n\n")
print(df.info(),end="\n\n#####################################################################################\n\n")
print(df.isnull().sum(),end="\n\n#####################################################################################\n\n")
print(df["is_fraud"].value_counts(),end="\n\n#####################################################################################\n\n")

"""
Veriler int ve float halinde EDA ya gerek yok o yüzden ve null değere sahip verin de yok o yüzden rahatız
sadece sıkıntı olabilcek nokta is_fraud sutununda 0 dan 9846 tane var ama 1 den sadece 154 tane var sayısı az
bakalım müdahale edecek miyiz
"""

########################################################################################################################

X = df.drop("is_fraud",axis=1)
y = df["is_fraud"]

sns.scatterplot(x = X["transaction_amount"],y = X["transaction_risk_score"],hue = y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

model = LogisticRegression()

penalty = ["l1","l2","elasticnet"] # ceza verme şekilleri
c_values = [100,10,1,0.1,0.01] # ceza verme kaysayıları
solver = ['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga']
class_weight = [{0:w,1:y} for w in [1,10,50,100] for y in [1,10,50,100]]

params = dict(penalty=penalty,solver=solver,class_weight=class_weight,C=c_values)

from sklearn.model_selection import GridSearchCV,StratifiedKFold

cv = StratifiedKFold()

grid = GridSearchCV(
    estimator=model,
    param_grid=params,
    scoring="accuracy",
    cv=cv,
)

import warnings
warnings.filterwarnings("ignore")

grid.fit(X_train,y_train)
y_pred = grid.predict(X_test)

score = accuracy_score(y_pred,y_test)
print(score,end="\n\n##############################################\n\n")
print(classification_report(y_pred,y_test),end="\n\n##############################################\n\n")
print(confusion_matrix(y_pred,y_test),end="\n\n##############################################\n\n")

# ROC & AUC

"""
ROC Eğrisi (Receiver Operating Characteristic Curve)

ROC Eğrisi, bir sınıflandırma modelinin farklı eşik (threshold) değerleri altındaki performansını görselleştiren bir grafiktir.
- Y ekseni: Doğru Pozitif Oranı (Recall)
- X ekseni: Yanlış Pozitif Oranı (1 - Özgüllük / Specificity)

Yorumlama
- Mükemmel model: Eğri, grafiğin sol üst köşesine ulaşır.
- Eğri sol üste ne kadar yakınsa, model o kadar iyidir.
- ROC Eğrisi altında kalan alan (AUC - Area Under Curve): Modelin başarısını sayısal olarak ifade eder.

AUC Değerleri
- AUC = 1 → Mükemmel sınıflandırıcı
- AUC = 0.5 → Rastgele tahmin (yazı tura atmak gibi)

Avantajı
- ROC eğrisi, özellikle dengesiz (imbalanced) veri setlerinde faydalıdır, çünkü modeli sınıf dağılımından veya seçilen eşikten bağımsız olarak değerlendirir.
"""

model_prob = grid.predict_proba(X_test) # probabilities for the positives (fraud) class
model_prob = model_prob[:,1]

from sklearn.metrics import roc_auc_score,roc_curve

model_auc = roc_auc_score(y_test,model_prob)

# Model False Pozitive Rate
# Model True Pozitive Rate
model_fpr , model_tpr , tresholds = roc_curve(y_test,model_prob)

plt.plot(model_fpr,model_tpr,label="Logistic",marker = ".")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()