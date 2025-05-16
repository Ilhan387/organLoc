import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from joblib import dump
import glob

ORGANS_DICT = {1247 : "trachea",
1302:    "right lung",
1326:    "left lung",
170:     "pancreas",
187:     "gallbladder",
237:     "urinary bladder",
2473:   "sternum",
29193:  "first lumbar vertebra",
29662:  "right kidney",
29663:  "left kidney",
30324:  "right adrenal gland",
30325:  "left adrenal gland",
32248:  "right psoas major",
32249:  "left psoas major",
40357:  "muscle body of right rectus abdominis",
40358:  "muscle body of left rectus abdominis",
480:    "aorta",
58:     "liver",
7578:   "thyroid gland",
86:     "spleen",
0:      "background",
1:      "body envelope",
2:      "thorax-abdomen"
}

def load_from_list(file_list):
    X_all, y_all = [], []
    for file in file_list:
        df = pd.read_csv(file, header=None)
        X_all.append(df.iloc[:, 1:].values)
        y_all.append(df.iloc[:, 0].values)
    return np.vstack(X_all), np.hstack(y_all)

# Matrice de confusion
def plot_clean_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[ORGANS_DICT[i] for i in labels])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', colorbar=True)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Label prédit", fontsize=12)
    ax.set_ylabel("Label réel", fontsize=12)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.show()