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
from fonctions import *

csv_files = sorted(glob.glob("../datas/CTce_ThAb_b33x33_n1000_8bit/*.csv"))
train_index_exp_2 = [ 0,  1,  2,  4,  6,  7,  8,  9, 10, 12, 13, 14, 15, 17, 19]
val_index_exp_2 = [3, 5, 11, 16, 18]

train_files_exp_2 = [csv_files[i] for i in train_index_exp_2]
val_files_exp_2 = [csv_files[i] for i in val_index_exp_2]


"""
    Création des vecteurs pour l'entrainement et la validation du modèle
"""

X_train, y_train = load_from_list(train_files_exp_2)
X_val, y_val = load_from_list(val_files_exp_2)

"""
    Entrainement du modèle et affichage des performances sur le val_set
"""

# 1. Paramètres optimisés
best_params = {
    'pca__n_components': 25,
    'knn__n_neighbors': 3,
    'knn__weights': 'distance'
}

# 2. Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('knn', KNeighborsClassifier())
])
pipeline.set_params(**best_params)

# 3. Entraînement sur le set d'entraînement
pipeline.fit(X_train, y_train)

# 4. Prédictions sur le set de validation
y_pred = pipeline.predict(X_val)

# Rapport de classification
print(f"\n=== Évaluation du modèle sur le set de validation ===")
print(classification_report(y_val, y_pred, target_names=[ORGANS_DICT[i] for i in np.unique(y_val)]))

# Labels uniques présents dans les données
labels = np.unique(np.concatenate([y_val, y_pred]))
print(f"\n=== Matrice de confusion pour le modèle ===")
plot_clean_confusion_matrix(y_val, y_pred, labels, "Matrice de confusion - Validation")
dump(pipeline, '../models/modelExp2.joblib')


