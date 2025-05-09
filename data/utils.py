import numpy as np

def scale_meg(meg_data, robust_scaler_center, robust_scaler_scale, sfreq, threshold=5):
    # Scale and center the data such that [-1, 1] is in IQR [0.25, 0.75]
    meg_data -= robust_scaler_center[:, None]
    meg_data /= robust_scaler_scale[:, None]

    # Clamp outliers above +threshold and below -threshold
    meg_data[np.abs(meg_data) > threshold] = (
        np.sign(meg_data[np.abs(meg_data) > threshold]) * threshold
    )

    # Apply baseline correction using the first 0.5 seconds of data
    meg_data -= np.mean(meg_data[..., :round(0.5 * sfreq)], axis=-1, keepdims=True)

    return meg_data