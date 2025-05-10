import numpy as np
import pandas as pd

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

def get_armeni_events(bids_root, subject, session, task):
    events_path = f"{bids_root}/sub-{subject}/ses-{session}/meg/sub-{subject}_ses-{session}_task-{task}_events.tsv"
    events_df = pd.read_csv(events_path, sep="\t")
    return events_df

def get_gwilliams_events(bids_root, subject, session, task):
    events_path = f"{bids_root}/sub-{subject}/ses-{session}/meg/sub-{subject}_ses-{session}_task-{task}_events.tsv"
    events_df = pd.read_csv(events_path, sep="\t")
    return events_df

def get_armeni_speech_events(bids_root, subject, session, task, sample_freq, duration, recording_samples):
    events_df = get_armeni_events(bids_root, subject, session, task)

    # Find all gaps between words
    speech_events = events_df[events_df["type"].str.contains("word_onset")]

    # Mark all time points as either speech or silence
    labels = np.zeros(recording_samples)
    for _, event in speech_events.iterrows():
        # Decision rule: if event is an "sp" mark it explicitly as silence
        onset = float(event["onset"])
        event_duration = float(event["duration"])
        t_start = (
            int(onset * sample_freq)
        )  # Delay labels so they occur at same time as brain response
        t_end = int((onset + event_duration) * sample_freq)

        labels[t_start : t_end + 1] = 0.0 if event["value"] == "sp" else 1.0
    
    # Compute frames of duration t seconds and decide if they are speech or silence by if they are > 50% speech
    samples = []
    for i in range(0, len(labels), round(sample_freq * duration)):
        if np.sum(labels[i : i + round(sample_freq * duration)]).mean() > 0.5:
            samples.append({
                "onset": i,
                "label": 1,
            })
        else:
            samples.append({
                "onset": i,
                "label": 0,
            })
    
    return samples

def get_gwilliams_speech_events(bids_root, subject, session, task, sample_freq, duration, recording_samples):
    events_df = get_gwilliams_events(bids_root, subject, session, task)

    # Find all gaps between words
    speech_events = events_df[
        ["'kind': 'word'" in trial_type for trial_type in list(events_df["trial_type"])]
    ]

    # Mark all time points as either speech or silence
    labels = np.zeros(recording_samples)
    for _, event in speech_events.iterrows():
        # Decision rule: if event is an "sp" mark it explicitly as silence
        onset = float(event["onset"])
        event_duration = float(event["duration"])
        t_start = (
            int(onset * sample_freq)
        )  # Delay labels so they occur at same time as brain response
        t_end = int((onset + event_duration) * sample_freq)

        labels[t_start : t_end + 1] = 1.0
    
    # Compute frames of duration t seconds and decide if they are speech or silence by if they are > 50% speech
    samples = []
    for i in range(0, len(labels), round(sample_freq * duration)):
        if np.sum(labels[i : i + round(sample_freq * duration)]).mean() > 0.5:
            samples.append({
                "onset": i,
                "label": 1,
            })
        else:
            samples.append({
                "onset": i,
                "label": 0,
            })
    
    return samples
