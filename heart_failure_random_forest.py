# THIS CODE IS SUBMITTED BY ayan-cs #

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

heart = pd.read_csv('heart_failure.csv')

X = np.array([heart.loc[i][:-1] for i in range(len(heart))])
y = np.array(heart['DEATH_EVENT'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print(f"Train Score : {clf.score(X_train, y_train)} \t Validation Score : {clf.score(X_test, y_test)}")

y_pred_prob = clf.predict_proba(X_test)
from sklearn.metrics import roc_auc_score, roc_curve
fpr0, tpr0, _ = roc_curve(y_test, y_pred_prob[:,1])

plt.plot(fpr0, tpr0)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print(f"AUC Score : {roc_auc_score(y_test, y_pred_prob[:,1])}")

diabetes_death = 0
diabetes_notdeath = 0
notdiabetes_death = 0

for i in range(len(heart)):
    if heart['diabetes'][i]==1 and y[i]==1:
        diabetes_death += 1
    elif heart['diabetes'][i]==1 and y[i]==0:
        diabetes_notdeath += 1
    elif heart['diabetes'][i]==0 and y[i]==1:
        notdiabetes_death += 1

print(f"Patients deceased having diabetes : {diabetes_death}\nPatients deceased not having Diabetes : {notdiabetes_death}\nPatients survived with diabetes : {diabetes_notdeath}")
