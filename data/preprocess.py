# Preprocesses data for Armeni, Gwilliams, Schoffelen, and CamCAN
import mne
import mne_bids
import numpy as np
import os
import h5py

from sklearn import preprocessing


def preprocess_data(dataset_name, config):
    """
    Preprocesses data for the specified dataset.

    Args:
        dataset_name (str): The name of the dataset to preprocess.
        bids_root (str): The root directory of the BIDS dataset.
        preproc_root (str): The root directory for preprocessed data.
        preproc_config (dict): Configuration for preprocessing.

    Returns:
        None
    """
    # Check if the dataset is supported
    if dataset_name not in ['Armeni', 'Gwilliams', 'Schoffelen', 'CamCAN']:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    # Perform preprocessing based on the dataset
    if dataset_name == 'Armeni':
        preprocess_armeni(dataset_config)
    elif dataset_name == 'Gwilliams':
        preprocess_gwilliams(dataset_config)
    elif dataset_name == 'Schoffelen':
        preprocess_schoffelen(dataset_config)
    elif dataset_name == 'CamCAN':
        preprocess_camcan(dataset_config)


def preprocess_armeni(config):
    """
    Preprocesses the Armeni dataset.

    Args:
        bids_root (str): The root directory of the BIDS dataset.
        preproc_root (str): The root directory for preprocessed data.
        config (dict): Configuration for preprocessing.

    Returns:
        None
    """

    preproc_root = config["preproc_root"]
    bids_root = config["bids_root"]

    for split in ["train", "val", "test"]:

        recordings = config["recordings"][split]

        for recording in recordings:

            subject = recording["subject"]
            sessions = recording["session"]
            task = recording["task"]

            for session in sessions:

                # Check if recording is already preprocessed
                preproc_path = f"{preproc_root}/{split}/sub-{subject}_ses-{session}_task-{task}_preproc.h5"

                if not os.path.exists(preproc_path):

                    print("Preprocessed data not found.")
                    print(
                        f"Preprocessing subject {subject} session {session} task {task}"
                    )

                    bids_path = mne_bids.BIDSPath(
                        subject=recording["subject"],
                        session=recording["session"],
                        task=recording["task"],
                        root=bids_root,
                    )

                    raw = mne_bids.read_raw_bids(bids_path, verbose=False)

                    # Filter out reference and other channels
                    channel_names = raw.info["ch_names"]
                    filtered_names = filtered_names = [
                        name for name in channel_names if name.startswith("M")
                    ]
                    raw = raw.pick(filtered_names)
                    # The dataset doesn't set a valid BIDS channel type (meggrad) so we set them as mag.
                    raw.set_channel_types(dict(zip(filtered_names, ["mag"] * len(filtered_names))))

                    raw = preprocess_meg(
                        raw,
                        resample_freq=config["resample_freq"],
                        l_freq=config["l_freq"],
                        h_freq=config["h_freq"],
                        notch_freq=config["notch_freq"],
                    )
                    data = raw.get_data()

                    sensor_positions = []
                    for ch in raw.info["chs"]:
                        pos = ch["loc"][:3]
                        sensor_positions.append(pos.tolist())


                    print("[+] Fitting robust scaler...")
                    robust_scaler = preprocessing.RobustScaler()
                    robust_scaler = robust_scaler.fit(data.transpose(1, 0)) # [S, T] -> [T, S]
                    print("[+] Scaler fitted.")

                    info = {
                        "subject": subject,
                        "session": session,
                        "subject_idx": int(subject) - 1,
                        "task": task,
                        "run": None,
                        "dataset": "armeni2022",
                        "sfreq": config["resample_freq"],
                        "sensor_xyz": sensor_positions,
                        "robust_scaler_center": robust_scaler.center_,
                        "robust_scaler_scale": robust_scaler.scale_,
                        "n_samples": data.shape[-1],
                    }

                    info["channel_means"] = np.mean(data, axis=1)
                    info["channel_stds"] = np.std(data, axis=1)

                    os.makedirs(preproc_path, exist_ok=True)
                    with h5py.File(preproc_path, "w") as f:
                        ds = f.create_dataset("data", data=data, dtype=np.float32, chunks=(data.shape[0], 40))
                        for key, value in info.items():
                            if value is None:
                                continue
                            ds.attrs[key] = value
                    
                    print("Finished preprocessing.")


def preprocess_meg(raw, resample_freq, l_freq, h_freq, notch_freq):
    # Band-pass filter the data to remove low and high frequency noise
    raw.load_data()
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks="all", n_jobs=-1, verbose=False)

    if h_freq > notch_freq:
        # Filter electric grid frequency and any harmonics if present in the signal
        raw.notch_filter(
            freqs=list(range(notch_freq, h_freq + 1, notch_freq)), verbose=False
        )

    # Decimate the signal by resampling (after cleaning up the signal already)
    raw.resample(sfreq=resample_freq, verbose=False)

    print("Preprocessed MEG. New sample rate", raw.info["sfreq"])

    return raw