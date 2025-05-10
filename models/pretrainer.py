# Lightning module for pretraining a model
import random
import lightning as L
import torch
import torchaudio

from .backbone import Backbone

class BandPretext(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        self.bands = [
            (0.1, 4.0),  # Delta
            (4.0, 8.0),  # Theta
            (8.0, 12.0),  # Alpha
            (12.0, 30.0),  # Beta
            (30.0, 70.0),  # Gamma
            (70.0, 100.0),  # Lower High Gamma
            (100.0, 125.0),  # Higher High Gamma 
        ]

        self.classifier = torch.nn.Linear(
            in_features=input_dim,
            out_features=len(self.bands),
        )
    
    def reject_band(self, x, sfreq):

        rejected_band_idx = random.randrange(len(self.bands))

        low_cutoff, high_cutoff = self.bands[rejected_band_idx]

        # Create a bandpass filter
        central_freq = (low_cutoff + high_cutoff) / 2
        bandwidth = high_cutoff - low_cutoff
        q_factor = central_freq / bandwidth

        filtered_x = torchaudio.functional.bandreject_biquad(
            x,
            sample_rate=sfreq,
            center_freq=central_freq,
            q=q_factor,
        )

        return filtered_x, rejected_band_idx
    
    def forward(self, filtered_x, rejected_band_idx):
        z = self.classifier(filtered_x)

        label = torch.full(
            size=(filtered_x.shape[0],),
            fill_value=rejected_band_idx,
            dtype=torch.long,
            device=filtered_x.device,
        )

        loss = torch.nn.functional.cross_entropy(z, label)
        probs = torch.nn.functional.softmax(z, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        return loss, preds



class Pretrainer(L.LightningModule):
    def __init__(
            self, input_dim, model_dim, subject_embedding_dim, flat_dim, projector_dim, learning_rate
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.backbone = Backbone(
            input_dim=input_dim,
            model_dim=model_dim,
            subject_embedding_dim=subject_embedding_dim,
        )

        self.ssl_projector = torch.nn.Sequential(
            torch.nn.Linear(flat_dim, projector_dim),
            torch.nn.GELU(),
            torch.nn.Linear(projector_dim, flat_dim),
        )

        self.band_pretext = BandPretext(input_dim=flat_dim)

    def forward(self, batch):

        meg = batch["meg"]
        subject_id = batch["subject_id"]
        dataset_id = batch["dataset_id"]
        sfreq = batch["sfreq"]

        # Apply band task
        filtered_x, rejected_band_idx = self.band_pretext.reject_band(meg.clone(), sfreq)
        z = self.backbone(filtered_x, dataset_id, subject_id)
        z = z.flatten(start_dim=1)
        z = self.ssl_projector(z)
        loss, preds = self.band_pretext(z, rejected_band_idx)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
        )