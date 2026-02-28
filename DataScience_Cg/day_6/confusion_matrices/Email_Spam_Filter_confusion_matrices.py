import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# True labels
y_true = np.array([1,0,1,0,1,0,1,0,1,0])

# Predictions
y_pred = np.array([1,0,1,0,0,0,1,1,1,0])

# Probabilities
y_prob = np.array([0.95,0.10,0.88,0.20,0.40,0.30,0.91,0.65,0.85,0.05])


# =========================
# Metrics
# =========================

print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1 Score :", f1_score(y_true, y_pred))
print("ROC-AUC  :", roc_auc_score(y_true, y_prob))


# =========================
# Confusion Matrix Plot
# =========================

cm = confusion_matrix(y_true, y_pred)

sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Purples',
            xticklabels=['Not Spam', 'Spam'],
            yticklabels=['Not Spam', 'Spam'])

plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()


 

# =========================

fpr, tpr, thresholds = roc_curve(y_true, y_prob)

plt.plot(fpr, tpr)
plt.plot([0,1], [0,1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()