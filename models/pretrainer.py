# Lightning module for pretraining a model
import lightning as L
import torch

from torchmetrics import Accuracy

from .backbone import Backbone

from .pretext.band import BandPretext
from .pretext.amp import AmpPretext
from .pretext.phase import PhasePretext

class Pretrainer(L.LightningModule):
    def __init__(
            self, input_dim, model_dim, subject_embedding_dim, projector_dim, learning_rate, datasets,
            tasks,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.tasks = tasks
        self.datasets = datasets

        self.backbone = Backbone(
            input_dim=input_dim,
            model_dim=model_dim,
            subject_embedding_dim=subject_embedding_dim,
        )

        self.ssl_projector = torch.nn.Sequential(
            torch.nn.Linear(model_dim + subject_embedding_dim, projector_dim),
            torch.nn.GELU(),
            torch.nn.Linear(projector_dim, model_dim),
        )

        self.band_pretext = BandPretext(input_dim=model_dim)
        self.amp_pretext = AmpPretext(input_dim=model_dim)
        self.phase_pretext = PhasePretext(input_dim=model_dim)

        self.metrics = {}
        for task in ["band", "amp", "phase"]:
            for split in ["train", "val", "test"]:

                if task == "band":
                    num_classes = len(self.band_pretext.bands)
                elif task == "amp":
                    num_classes = self.amp_pretext.num_steps
                elif task == "phase":
                    num_classes = self.phase_pretext.num_steps

                self.metrics[f"{split}_acc_{task}"] = Accuracy(
                    task="multiclass",
                    average="macro",
                    num_classes=num_classes,
                )
        self.metrics = torch.nn.ModuleDict(self.metrics)

    def forward(self, batch, split):

        meg = batch["meg"]
        subject_id = batch["subject_id"]
        dataset_id = batch["dataset_id"]
        sfreq = batch["sfreq"]

        if "band" in self.tasks:
            # Apply band task
            filtered_x, rejected_band_idx = self.band_pretext.reject_band(meg.clone(), sfreq)
            z = self.backbone(filtered_x, dataset_id, subject_id)
            z = self.ssl_projector(z)
            loss_band, preds_band, targets_band = self.band_pretext(z, rejected_band_idx)
            self.log(f"{split}_loss_band", loss_band)
            self.metrics[f"{split}_acc_band"](preds_band, targets_band)
            self.log(f"{split}_acc_band", self.metrics[f"{split}_acc_band"])
        else:
            loss_band = torch.tensor(0.0, device=self.device, requires_grad=True)

        if "amp" in self.tasks:
            # Apply amplitude task
            scaled_x, scale_label = self.amp_pretext.scale_amp(meg.clone())
            z = self.backbone(scaled_x, dataset_id, subject_id)
            z = self.ssl_projector(z)
            loss_amp, preds_amp, targets_amp = self.amp_pretext(z, scale_label)
            self.log(f"{split}_loss_amp", loss_amp)
            self.metrics[f"{split}_acc_amp"](preds_amp, targets_amp)
            self.log(f"{split}_acc_amp", self.metrics[f"{split}_acc_amp"])
        else:
            loss_amp = torch.tensor(0.0, device=self.device, requires_grad=True)

        if "phase" in self.tasks:
            # Apply phase task
            phase_x, phase_label = self.phase_pretext.phase_shift(meg.clone())
            z = self.backbone(phase_x, dataset_id, subject_id)
            z = self.ssl_projector(z)
            loss_phase, preds_phase, targets_phase = self.phase_pretext(z, phase_label)
            self.log(f"{split}_loss_phase", loss_phase)
            self.metrics[f"{split}_acc_phase"](preds_phase, targets_phase)
            self.log(f"{split}_acc_phase", self.metrics[f"{split}_acc_phase"])
        else:
            loss_phase = torch.tensor(0.0, device=self.device, requires_grad=True)

        loss = loss_band + loss_amp + loss_phase

        return loss

    def training_step(self, batch, batch_idx):
        loss = self(batch, split="train")
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self(batch, split="val")
        self.log("val_loss", loss)
    
    def test_step(self, batch, batch_idx):
        loss = self(batch, split="test")
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
        )