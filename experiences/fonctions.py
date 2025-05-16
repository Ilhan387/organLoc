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

# Exemple d'utilisation

def extract_ids(file_paths):
    """
    Extrait les identifiants (IDs) depuis une liste de chemins de fichiers .nii.gz.

    Paramètre :
        file_paths (list): Liste de chemins de fichiers.

    Retour :
        list: Liste des IDs extraits sous forme d'entiers.
    """
    ids = []
    for path in file_paths:
        match = re.search(r'(\d+)_\d+_CTce_ThAb\.nii\.gz', path)
        if match:
            ids.append(int(match.group(1)))
    return ids


def scale_to_255(volume, in_min=-1023, in_max=2976, out_min=0, out_max=255):
    """
    Mise à l'échelle linéaire d'un volume 3D vers l'intervalle [0, 255].

    Args:
        volume (np.ndarray): tableau 3D contenant les valeurs à transformer.
        in_min (float): borne inférieure des valeurs d'entrée.
        in_max (float): borne supérieure des valeurs d'entrée.
        out_min (float): borne inférieure des valeurs de sortie.
        out_max (float): borne supérieure des valeurs de sortie.

    Returns:
        np.ndarray: volume 3D mis à l’échelle sur [0, 255], de type uint8.
    """
    scaled = (volume - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    scaled = np.clip(scaled, out_min, out_max)  # Pour éviter les débordements
    return scaled.astype(np.uint8)

def voxel_to_ras(voxel_coord, affine):
    """
    Chuyển từ voxel (i, j, k) sang RAS+ bằng affine matrix.
    """
    voxel_coord = np.append(voxel_coord, 1)  # Thêm phần đồng nhất
    ras_coord = affine @ voxel_coord
    return ras_coord[:3]

def extract_axial_patches_with_positions(volume, img_affine,patch_size=33, stride=1):
    assert patch_size % 2 == 1, "patch_size must be odd"
    half = patch_size // 2

    Z, Y, X = volume.shape  # ✅ bon ordre

    axial_patches = []
    axial_positions = []

    for z in range(half, Z - half, stride):
        for y in range(half, Y - half, stride):
            for x in range(half, X - half, stride):
                patch = volume[z, y - half : y + half + 1, x - half : x + half + 1]  # ✅ coupe XY à z
                patch_flat = patch.ravel()
                axial_patches.append(patch_flat)
                axial_positions.append(voxel_to_ras((z,y,x), img_affine))

    axial_patches = np.stack(axial_patches)
    axial_positions = np.array(axial_positions)

    return axial_patches, axial_positions

def compute_estimated_centers(pipeline, probas_patches, patches_positions):
    """
    Calcule les centres estimés pondérés pour chaque organe du classifieur.

    Args:
        pipeline: classifieur avec attribut `.classes_`
        probas_axial, probas_coronal, probas_sagittal: np.ndarray de shape (N, n_classes)
        axial_positions, coronal_positions, sagittal_positions: np.ndarray de shape (N, 3)

    Returns:
        dict: {organe_id: (z, y, x)} centre estimé
    """
    estimated_centers = {}

    for organ_id in pipeline.classes_:
        index = np.where(pipeline.classes_ == organ_id)[0][0]

        # Récupère les probas pour l'organe considéré
        p = probas_patches[:, index]

        # Axial
        N = np.sum(p)
        xc = np.sum(patches_positions * p[:, None], axis=0) / N if N > 0 else np.array([0, 0, 0])

        estimated_center = xc
        estimated_centers[organ_id] = tuple(estimated_center)

    return {
        int(k): tuple(float(v_) for v_ in v)
        for k, v in estimated_centers.items()
    }

def load_true_centers(patient_id, folder_path):
    """
    Charge les centres vrais (ground truth) des organes pour un patient donné.

    Args:
        patient_id (str): identifiant du patient, ex: "10000135"
        folder_path (str): chemin vers le dossier contenant les fichiers *_center.csv

    Returns:
        dict: mapping {organe_id: (z, y, x)}
    """
    centers = {}

    for filename in os.listdir(folder_path):
        if filename.startswith(f"{patient_id}_") and filename.endswith("_center.csv"):
            # Extrait l'ID de l'organe depuis le nom du fichier
            parts = filename.split("_")
            try:
                organ_id = int(parts[-2])  # Exemple: "29662" dans "xxx_29662_center.csv"
            except ValueError:
                continue  # Skip si on n'arrive pas à parser l'ID

            filepath = os.path.join(folder_path, filename)

            with open(filepath, "r") as f:
                reader = csv.reader(f)
                row = next(reader)
                try:
                    x, y, z = map(float, row[3:])  # Prend les 3 premières valeurs
                    centers[organ_id] = (x, y, z)
                except:
                    print(f"Erreur de parsing dans le fichier {filename}")

    return centers

def remove_first_n_items(d, n):
    return dict(list(d.items())[n:])

def compute_errors(true_centers, estimated_centers):
    """
    Calcule l'erreur euclidienne entre les centres prédits et vrais.

    Args:
        true_centers (dict): dictionnaire {organe: (x, y, z)} réel
        estimated_centers (dict): dictionnaire {organe: (x, y, z)} estimé

    Returns:
        dict: {organe: erreur euclidienne}
    """
    errors = {}
    for organ in true_centers:
        if organ in estimated_centers:
            true = np.array(true_centers[organ])
            estimated = np.array(estimated_centers[organ])
            distance = np.linalg.norm(true - estimated)
            errors[organ] = distance
        else:
            print(f"⚠️ Organe {organ} manquant dans estimated_centers")

    return errors

def compute_stats_all_images(true_centers_list, estimated_centers_list, organ_dict):
    from collections import defaultdict
    all_errors = defaultdict(list)  # {organ_id: [erreur1, erreur2, ...]}

    for true_c, est_c in zip(true_centers_list, estimated_centers_list):
        errors = compute_errors(true_c, est_c)
        for organ_id, error in errors.items():
            all_errors[organ_id].append(error)

    stats = {}
    for organ_id, errors in all_errors.items():
        mean = np.mean(errors)
        std = np.std(errors)
        organ_name = organ_dict.get(organ_id, f"Organ {organ_id}")
        stats[organ_name] = (mean, std)
        print(f"{organ_name}: {mean:.2f} ± {std:.2f}")

    return stats

def plot_organ_distance_errors(stats_dict, title="Distance Error by Organ", run_label="Run"):
    organ_names = list(stats_dict.keys())
    means = [stats_dict[organ][0] for organ in organ_names]
    stds = [stats_dict[organ][1] for organ in organ_names]

    y_pos = np.arange(len(organ_names))

    plt.figure(figsize=(14, 8))
    plt.barh(y_pos, means, xerr=stds, align='center', color='skyblue', ecolor='black', capsize=4, label=run_label)
    plt.yticks(y_pos, organ_names)
    plt.xlabel('Distance Error (mm)')
    plt.title(title)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()