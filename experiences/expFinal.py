import matplotlib.pyplot as plt
import numpy as np

average_errors = {
    "liver": 42.235,
    "spleen": 106.925,
    "pancreas": 47.4175,
    "gallbladder": 94.2875,
    "urinary bladder": 156.9375,
    "aorta": 34.155,
    "trachea": 234.09,
    "right lung": 69.0075,
    "left lung": 90.6025,
    "sternum": 199.1825,
    "thyroid gland": 193.6475,
    "first lumbar vertebra": 68.7675,
    "right kidney": 81.8275,
    "left kidney": 77.44,
    "right adrenal gland": 54.3175,
    "left adrenal gland": 88.4825,
    "right psoas major": 76.145,
    "left psoas major": 72.4175,
    "muscle body of right rectus abdominis": 129.5225,
    "muscle body of left rectus abdominis": 80.5275
}

average_std_devs = {
    "liver": 17.8375,
    "spleen": 11.6025,
    "pancreas": 12.0575,
    "gallbladder": 16.105,
    "urinary bladder": 23.4225,
    "aorta": 10.63,
    "trachea": 32.215,
    "right lung": 9.9575,
    "left lung": 16.895,
    "sternum": 32.4575,
    "thyroid gland": 16.095,
    "first lumbar vertebra": 16.7575,
    "right kidney": 10.64,
    "left kidney": 12.3875,
    "right adrenal gland": 14.865,
    "left adrenal gland": 22.4625,
    "right psoas major": 21.26,
    "left psoas major": 21.485,
    "muscle body of right rectus abdominis": 20.705,
    "muscle body of left rectus abdominis": 12.3025
}

def plot_average_errors_with_std(average_errors, average_std_devs, title="Distance Error with Std Dev"):
    organs = list(average_errors.keys())
    means = np.array([average_errors[org] for org in organs])
    stds = np.array([average_std_devs[org] for org in organs])
    y_pos = np.arange(len(organs))

    plt.figure(figsize=(12, 8))
    plt.barh(y_pos, means, xerr=stds, align='center', color='skyblue', ecolor='black', capsize=5)
    plt.yticks(y_pos, organs)
    plt.xlabel('Distance Error (mm)')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

plot_average_errors_with_std(average_errors, average_std_devs)