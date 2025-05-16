from joblib import load
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import re
import glob
import sys
import os
import glob
from fonctions import *

files3D = sorted(glob.glob("../datas/CTce_ThAb/*.nii.gz"))

val_index_exp_1 = [0, 1, 8, 15, 17]

val_files_3D_exp_1 = [files3D[i] for i in val_index_exp_1]

# Chargement du modèle
pipeline = load('../models/modelExp1.joblib')


datas_val = []
images_affine = []


for file in (val_files_3D_exp_1):
    img = nib.load(file)
    affine = img.affine
    images_affine.append(affine)
    data = img.get_fdata()
    data_scaled = scale_to_255(data)
    datas_val.append(data_scaled)

val_patches = []
val_patches_positions = []

for img3D,img_affine in zip(datas_val,images_affine):
    patches, positions = extract_axial_patches_with_positions(img3D, img_affine,patch_size=33, stride = 10)
    val_patches.append(patches)
    val_patches_positions.append(positions)

val_patches_patient1 = val_patches[0]
positions_val_patches_patient1 = val_patches_positions[0]

val_patches_patient2 = val_patches[1]
positions_val_patches_patient2 = val_patches_positions[1]

val_patches_patient3 = val_patches[2]
positions_val_patches_patient3 = val_patches_positions[2]

val_patches_patient4 = val_patches[3]
positions_val_patches_patient4 = val_patches_positions[3]

val_patches_patient5 = val_patches[4]
positions_val_patches_patient5 = val_patches_positions[4]


probas_train_patches_patient1 = pipeline.predict_proba(val_patches_patient1)
probas_train_patches_patient2 = pipeline.predict_proba(val_patches_patient2)
probas_train_patches_patient3 = pipeline.predict_proba(val_patches_patient3)
probas_train_patches_patient4 = pipeline.predict_proba(val_patches_patient4)
probas_train_patches_patient5 = pipeline.predict_proba(val_patches_patient5)

classes = pipeline.classes_

patients_ids = extract_ids(val_files_3D_exp_1)

true_centers_patient_1 = dict(sorted(load_true_centers(patients_ids[0], "../datas/centers").items()))
true_centers_patient_2 = dict(sorted(load_true_centers(patients_ids[1], "../datas/centers").items()))
true_centers_patient_3 = dict(sorted(load_true_centers(patients_ids[2], "../datas/centers").items()))
true_centers_patient_4 = dict(sorted(load_true_centers(patients_ids[3], "../datas/centers").items()))
true_centers_patient_5 = dict(sorted(load_true_centers(patients_ids[4], "../datas/centers").items()))


estimated_centers_patient_1 = remove_first_n_items(compute_estimated_centers(pipeline, probas_train_patches_patient1, positions_val_patches_patient1),3)
estimated_centers_patient_2 = remove_first_n_items(compute_estimated_centers(pipeline, probas_train_patches_patient2, positions_val_patches_patient2),3)
estimated_centers_patient_3 = remove_first_n_items(compute_estimated_centers(pipeline, probas_train_patches_patient3, positions_val_patches_patient3),3)
estimated_centers_patient_4 = remove_first_n_items(compute_estimated_centers(pipeline, probas_train_patches_patient4, positions_val_patches_patient4),3)
estimated_centers_patient_5 = remove_first_n_items(compute_estimated_centers(pipeline, probas_train_patches_patient5, positions_val_patches_patient5),3)

estimated_centers_list = [estimated_centers_patient_1, estimated_centers_patient_2, 
                          estimated_centers_patient_3, estimated_centers_patient_4, estimated_centers_patient_5]

true_centers_list = [true_centers_patient_1, true_centers_patient_2, 
                          true_centers_patient_3, true_centers_patient_4, true_centers_patient_5]

stats = compute_stats_all_images(true_centers_list, estimated_centers_list, ORGANS_DICT)

plot_organ_distance_errors(stats, title="Résultats exp1", run_label="Run exp1")
