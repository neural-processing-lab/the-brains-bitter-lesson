# Preprocesses data for Armeni, Gwilliams, Schoffelen, and CamCAN
import glob
import mne
import mne_bids
import numpy as np
import os
import h5py

import constants

from sklearn import preprocessing

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
    if dataset_name not in ['armeni2022', 'gwilliams2022', 'mous', 'camcan']:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    
    if "debug" in config["preproc_root"]:
        return

    # Perform preprocessing based on the dataset
    if dataset_name == 'armeni2022':
        preprocess_armeni(config)
    elif dataset_name == "mous":
        preprocess_mous(config)
    elif dataset_name == 'camcan':
        preprocess_camcan(config)


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
            sessions = recording["sessions"]
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
                        session=session,
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

                    os.makedirs(f"{preproc_root}/{split}", exist_ok=True)
                    with h5py.File(preproc_path, "w") as f:
                        ds = f.create_dataset("data", data=data, dtype=np.float32, chunks=(data.shape[0], 40))
                        for key, value in info.items():
                            if value is None:
                                continue
                            ds.attrs[key] = value
                    
                    print("Finished preprocessing.")


def preprocess_camcan(config):
    """
    Preprocesses the CamCAN dataset.

    Args:
        bids_root (str): The root directory of the BIDS dataset.
        preproc_root (str): The root directory for preprocessed data.
        config (dict): Configuration for preprocessing.

    Returns:
        None
    """

    preproc_root = config["preproc_root"]
    bids_root = config["bids_root"]

    subject_no = 0
    subject_hashmap = {}

    for task in ["rest", "smt"]:

        # Find all subjects
        subjects = [
            os.path.basename(subject).replace("sub-", "") for subject in sorted(glob.glob(bids_root + f"/{task}/sub-*"))
        ]

        # Generate splits
        n_train = int(len(subjects) * 0.9)
        n_val = int(len(subjects) * 0.05)
        train_subjects = subjects[:n_train]
        val_subjects = subjects[n_train : n_train + n_val]
        test_subjects = subjects[n_train + n_val :]

        for split, subjects in zip(["train", "val", "test"], [train_subjects, val_subjects, test_subjects]):

            for subject in subjects:

                if not subject in subject_hashmap:
                    subject_hashmap[subject] = subject_no
                    subject_no += 1

                raw_path = bids_root + f"/{task}/sub-{subject}/ses-{task}/meg/sub-{subject}_ses-{task}_task-{task}_meg.fif"
                preproc_path = f"{preproc_root}/{split}/sub-{subject}_ses-{task}_task-{task}_preproc.h5"

                if not os.path.exists(preproc_path):

                    print("Preprocessed data not found.")
                    print(
                        f"Preprocessing subject {subject} session {task} task {task}"
                    )

                    raw = mne.io.read_raw_fif(raw_path, verbose=False)

                    channel_names = raw.info['ch_names']
                    filtered_names = filtered_names = [name for name in channel_names if name.startswith('MEG')]
                    raw = raw.pick(filtered_names)

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
                            "session": task,
                            "subject_idx": subject_hashmap[subject],
                            "task": task,
                            "run": None,
                            "dataset": "camcan",
                            "sfreq": config["resample_freq"],
                            "sensor_xyz": sensor_positions,
                            "robust_scaler_center": robust_scaler.center_,
                            "robust_scaler_scale": robust_scaler.scale_,
                            "n_samples": data.shape[-1],
                        }

                    info["channel_means"] = np.mean(data, axis=1)
                    info["channel_stds"] = np.std(data, axis=1)

                    os.makedirs(f"{preproc_root}/{split}", exist_ok=True)
                    with h5py.File(preproc_path, "w") as f:
                        ds = f.create_dataset("data", data=data, dtype=np.float32, chunks=(data.shape[0], 40))
                        for key, value in info.items():
                            if value is None:
                                continue
                            ds.attrs[key] = value
                    
                    print("Finished preprocessing.")


def preprocess_mous(config):
    """
    Preprocesses the Mous dataset.

    Args:
        bids_root (str): The root directory of the BIDS dataset.
        preproc_root (str): The root directory for preprocessed data.
        config (dict): Configuration for preprocessing.

    Returns:
        None
    """

    preproc_root = config["preproc_root"]
    bids_root = config["bids_root"]

    # Find all subjects
    subjects = [
        os.path.basename(subject).replace("sub-", "") for subject in sorted(glob.glob(bids_root + "/sub-*"))
    ]

    subject_hashmap = {subject: i for i, subject in enumerate(subjects)}

    # Generate splits
    n_train = int(len(subjects) * 0.84)
    n_val = int(len(subjects) * 0.08)
    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train : n_train + n_val]
    test_subjects = subjects[n_train + n_val :]

    for split, subjects in zip(["train", "val", "test"], [train_subjects, val_subjects, test_subjects]):

        for subject in subjects:

            if subject.startswith("A"):
                stim_task = "auditory"
            elif subject.startswith("V"):
                stim_task = "visual"

            tasks = ["rest", stim_task]

            for task in tasks:

                raw_path = bids_root + f"/sub-{subject}/meg/sub-{subject}_task-{task}_meg.ds"
                preproc_path = f"{preproc_root}/{split}/sub-{subject}_task-{task}_preproc.h5"

                if not os.path.exists(preproc_path):

                    print("Preprocessed data not found.")
                    print(
                        f"Preprocessing subject {subject} task {task}"
                    )

                    try:
                        raw = mne.io.read_raw_ctf(raw_path, verbose=False)
                    except Exception as e:
                        continue

                    channel_names = raw.info['ch_names']
                    filtered_names = [name for name in channel_names if name.split('-')[0] in constants.MOUS_CHANNELS]
                    raw = raw.pick(filtered_names)
                    channel_types = {ch: 'mag' for ch in filtered_names}
                    raw.set_channel_types(channel_types)

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
                            "session": None,
                            "subject_idx": subject_hashmap[subject],
                            "task": task,
                            "run": None,
                            "dataset": "mous",
                            "sfreq": config["resample_freq"],
                            "sensor_xyz": sensor_positions,
                            "robust_scaler_center": robust_scaler.center_,
                            "robust_scaler_scale": robust_scaler.scale_,
                            "n_samples": data.shape[-1],
                        }

                    info["channel_means"] = np.mean(data, axis=1)
                    info["channel_stds"] = np.std(data, axis=1)

                    os.makedirs(f"{preproc_root}/{split}", exist_ok=True)
                    with h5py.File(preproc_path, "w") as f:
                        ds = f.create_dataset("data", data=data, dtype=np.float32, chunks=(data.shape[0], 40))
                        for key, value in info.items():
                            if value is None:
                                continue
                            ds.attrs[key] = value
                    
                    print("Finished preprocessing.")