import glob
import h5py
import torch

from torch.utils.data.dataloader import default_collate
from abc import abstractmethod
from .preprocess import preprocess_data
from . import utils

GET_SPEECH_EVENTS_FN = {
    "armeni2022": utils.get_armeni_speech_events,
    "gwilliams2022": utils.get_gwilliams_speech_events,
}


class PaddingCollator:
    def __init__(self, max_pad, padding_key="meg"):
        self.padding_key = padding_key
        self.max_pad = max_pad
        
    def __call__(self, batch):
        # Extract your special field from each element
        special_items = [item[self.padding_key] for item in batch]
        
        # Handle your special field however you want
        special_batch = self._custom_collate(special_items)
        
        # Remove special field from each item temporarily
        for item in batch:
            item.pop(self.padding_key, None)
            
        # Run default collation on everything else
        regular_batch = default_collate(batch)
        
        # Put special field back in
        regular_batch[self.padding_key] = special_batch
        
        return regular_batch
    
    def _custom_collate(self, meg):
        # Right-pad sensor dimension to the length pad_features.
        padded_meg = torch.stack([
            torch.nn.functional.pad(
                sample, pad=(0, 0, 0, self.max_pad - sample.shape[0])
            ) for sample in meg
        ])

        return padded_meg

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
    def __init__(self, dataset_name, datasets_config, split, sample_duration=0.5, dataset_id=0, subject_id_increment=0):
        super().__init__(dataset_name, datasets_config)

        self.sample_duration = sample_duration
        self.dataset_id = dataset_id
        self.subject_id_increment = subject_id_increment

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

            speech_events = GET_SPEECH_EVENTS_FN[dataset_name](
                bids_root=self.config["bids_root"],
                subject=subject,
                session=session,
                task=task,
                sample_freq=sfreq,
                duration=self.sample_duration,
                recording_samples=info["n_samples"],
            )
        
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
        meg = utils.scale_meg(meg, robust_scaler_center, robust_scaler_scale, sfreq)

        return {
            "meg": meg,
            "dataset_id": self.dataset_id,
            "subject_id": subject_id + self.subject_id_increment,
            "sensor_xyz": sensor_xyz,
            "onset": onset,
            "speech": label,
            "sfreq": sfreq,
        }

# class VoicingDataset(MEGDataset):
#     def __init__(self, data, labels):
#         self.data = data
#         self.labels = labels

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx], self.labels[idx]

class PretrainingDataset(MEGDataset):
    def __init__(self, dataset_name, datasets_config, split, sample_duration=0.5, dataset_id=0, subject_id_increment=0):
        super().__init__(dataset_name, datasets_config)

        self.sample_duration = sample_duration
        self.dataset_id = dataset_id
        self.subject_id_increment = subject_id_increment

        self.samples = []
        
        preprocessed_recording_paths = sorted(glob.glob(self.config["preproc_root"] + f"/{split}/*.h5"))
        self.preprocessed_recordings = [h5py.File(path, "r") for path in preprocessed_recording_paths]

        for preprocessed_recording in self.preprocessed_recordings:

            info = dict(preprocessed_recording.attrs)
            sfreq = info["sfreq"]
            recording_samples = info["n_samples"]

            for i in range(0, recording_samples, round(sfreq * self.sample_duration)):

                if i + round(sfreq * self.sample_duration) > recording_samples:
                    break

                self.samples.append({
                    "recording": preprocessed_recording,
                    "onset": i,
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

        meg = sample["recording"][..., onset : onset + round(self.sample_duration * sfreq)]
        meg = utils.scale_meg(meg, robust_scaler_center, robust_scaler_scale, sfreq)

        return {
            "meg": meg,
            "dataset_id": self.dataset_id,
            "subject_id": subject_id + self.subject_id_increment,
            "sensor_xyz": sensor_xyz,
            "onset": onset,
            "sfreq": sfreq,
        }