# Lightning module for probing a model
import lightning as L
import torch

from torchmetrics import Accuracy, AUROC

class Prober(L.LightningModule):
    def __init__(self, backbone, model_dim, subject_embedding_dim, learning_rate, use_mean_subject_embedding=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.use_mean_subject_embedding = use_mean_subject_embedding

        self.backbone = backbone
        # Freeze the backbone's encoder only (leaving dataset and subject parameters trainable)
        for param in self.backbone.encoder.parameters():
            param.requires_grad = False

        self.classifier = torch.nn.Linear(
            in_features=model_dim + subject_embedding_dim,
            out_features=1,
        )

        self.metrics = {}
        for split in ["train", "val", "test"]:
            self.metrics[f"{split}_acc"] = Accuracy(
                task="multiclass",
                average="macro",
                num_classes=2,
            )
            self.metrics[f"{split}_auroc"] = AUROC(
                task="binary",
            )
        self.metrics = torch.nn.ModuleDict(self.metrics)
    
    def forward(self, batch, split):
        meg = batch["meg"]
        subject_id = batch["subject_id"]
        dataset_id = batch["dataset_id"]
        label = batch["label"]

        z = self.backbone(
            meg,
            dataset_id=dataset_id,
            subject_id=subject_id,
            use_mean_subject_embedding=self.use_mean_subject_embedding,
        )

        logits = self.classifier(z).squeeze(-1)
        probs = torch.sigmoid(logits)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            label.float(),
        )

        # Compute accuracy
        self.metrics[f"{split}_acc"](probs, label)
        self.metrics[f"{split}_auroc"](probs, label)

        self.log(
            f"{split}_acc",
            self.metrics[f"{split}_acc"],
        )
        self.log(
            f"{split}_auroc",
            self.metrics[f"{split}_auroc"],
        )

        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self(batch, split="train")
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self(batch, split="val")
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self(batch, split="test")
        self.log("test_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )