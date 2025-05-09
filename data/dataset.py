import glob
import h5py
import torch

from abc import abstractmethod
from .preprocess import preprocess_data
from .utils import scale_meg

class MEGDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, datasets_config):
        self.dataset_name = dataset_name
        self.config = datasets_config[self.dataset_name]
        
        preprocess_data(self.dataset_name, self.config)

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

class SpeechDataset(MEGDataset):
    def __init__(self, dataset_name, datasets_config, split, sample_duration=0.5):
        super().__init__(dataset_name, datasets_config)

        self.sample_duration = sample_duration

        self.samples = []
        
        preprocessed_recording_paths = sorted(glob.glob(self.config["preproc_root"] + f"/{split}/*.h5"))
        self.preprocessed_recordings = [h5py.File(path, "r") for path in preprocessed_recording_paths]

        for preprocessed_recording in self.preprocessed_recordings:

            info = dict(preprocessed_recording.attrs)
            subject = info["subject"]
            session = info["session"]
            task = info["task"]
            run = info["run"]
            sfreq = info["sfreq"]

            # Get labels for this recording
            raise NotImplementedError("Label extraction not implemented yet")
        
            for speech_event in speech_events:
                self.samples.append({
                    "recording": preprocessed_recording,
                    "onset": speech_event["onset"],
                    "label": speech_event["label"],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        subject_id = sample["recording"].attrs["subject_idx"]
        robust_scaler_center = sample["recording"].attrs["robust_scaler_center"]
        robust_scaler_scale = sample["recording"].attrs["robust_scaler_scale"]
        sfreq = sample["recording"].attrs["sfreq"]
        sensor_xyz = sample["recording"].attrs["sensor_xyz"]

        onset = sample["onset"]
        label = sample["label"]

        meg = sample["recording"][..., onset : onset + round(self.sample_duration * sfreq)]
        meg = scale_meg(meg, robust_scaler_center, robust_scaler_scale, sfreq)

        return {
            "meg": meg,
            "subject_id": subject_id,
            "sensor_xyz": sensor_xyz,
            "onset": onset,
            "speech": label,
        }

# class VoicingDataset(MEGDataset):
#     def __init__(self, data, labels):
#         self.data = data
#         self.labels = labels

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx], self.labels[idx]

# class PretrainingDataset(MEGDataset):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx], self.labels[idx]