import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
import builtins
pd.set_option("display.max_columns", None)

def print(*args, **kwargs):
    # Eğer end parametresi belirtilmemişse, otomatik ayarla
    if "end" not in kwargs:
        kwargs["end"] = "\n\n###############################################\n\n"   # Buraya istediğin varsayılanı koy
    return builtins.print(*args, **kwargs)

# ilk olarak Email Classification
"""
subject_formality_score --> ne kadar resmi yazılıdığının skoru 
sender_relationship_score -->  yollayan kişinin yakınlık skoru
email_type --> 0 : Personal , 1 : Work Email
"""

"""

#                   BASİT DATASET

df = pd.read_csv("9-email_classification_svm.csv")
print(df,end="\n\n###############################################\n\n")

sns.scatterplot(x=df["subject_formality_score"] , y=df["sender_relationship_score"],hue=df["email_type"])
plt.show()

X = df.drop("email_type",axis = 1)
Y = df["email_type"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 15)

from sklearn.svm import SVC

svc = SVC(kernel="linear")
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_pred,y_test),end="\n\n##############################################\n\n")
print(confusion_matrix(y_pred,y_test),end="\n\n##############################################\n\n")
"""

"""
#                    ORTA DATASET VE CSV
df = pd.read_csv("9-loan_risk_svm.csv")

print(df)

X = df.drop("loan_risk",axis=1)
y = df["loan_risk"]

sns.scatterplot(x=X["credit_score_fluctuation"],y=X["recent_transaction_volume"],hue=y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 15)

####################################################

linear = SVC(kernel="linear")
linear.fit(X_train, y_train)
y_pred_svc = linear.predict(X_test)

print(classification_report(y_pred_svc,y_test))
print(confusion_matrix(y_pred_svc,y_test))

####################################################

rbf = SVC(kernel="rbf")
rbf.fit(X_train, y_train)
y_pred_rbf = rbf.predict(X_test)

print(classification_report(y_pred_rbf,y_test))
print(confusion_matrix(y_pred_rbf,y_test))

####################################################

poly = SVC(kernel="poly")
poly.fit(X_train, y_train)
y_pred_poly = poly.predict(X_test)

print(classification_report(y_pred_poly,y_test))
print(confusion_matrix(y_pred_poly,y_test))

####################################################

sigmoid = SVC(kernel="sigmoid")
sigmoid.fit(X_train, y_train)
y_pred_sigmoid = sigmoid.predict(X_test)

print(classification_report(y_pred_sigmoid,y_test))
print(confusion_matrix(y_pred_sigmoid,y_test))

####################################################

# HYPERPARAMETER TUNİNG

param_grid = {
    "C":[0.1,1,10,100,1000],
    "kernel":["rbf"],
    "gamma" : ["scale","auto"]
}

grid = GridSearchCV(SVC(), param_grid=param_grid, n_jobs=-1,cv = 5)
grid.fit(X_train, y_train)
y_pred_hiper = grid.predict(X_test)
print(classification_report(y_pred_hiper,y_test))
print(confusion_matrix(y_pred_hiper,y_test))
"""


df = pd.read_csv("9-seismic_activity_svm.csv")
print(df)

"""
#sns.scatterplot(x=df["underground_wave_energy"],y=df["vibration_axis_variation"],hue=df["seismic_event_detected"])
#plt.show()

# MANUEL RBF KERNEL

df["underground_wave_energy ** 2"] = df["underground_wave_energy"]**2
df["vibration_axis_variation ** 2"] = df["vibration_axis_variation"]**2
df["underground_wave_energy * vibration_axis_variation"] = df["underground_wave_energy"] * df["vibration_axis_variation"]

X = df.drop("seismic_event_detected",axis=1)
y = df["seismic_event_detected"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 15)

import plotly.express as px

#fig = px.scatter_3d(df, x="underground_wave_energy ** 2", y="vibration_axis_variation ** 2",z = "underground_wave_energy * vibration_axis_variation",color="seismic_event_detected")
#wfig.show()

linear = SVC(kernel="linear")
linear.fit(X_train, y_train)
y_pred_linear = linear.predict(X_test)

print(classification_report(y_pred_linear,y_test))
print(confusion_matrix(y_pred_linear,y_test))
"""


# AUTO RBF

X = df.drop("seismic_event_detected",axis=1)
y = df["seismic_event_detected"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 15)

rbf = SVC(kernel="rbf")
rbf.fit(X_train, y_train)
y_pred_linear = rbf.predict(X_test)
print(classification_report(y_pred_linear,y_test))
print(confusion_matrix(y_pred_linear,y_test))
