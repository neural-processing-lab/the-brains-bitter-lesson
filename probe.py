# Given a pre-trained model, probe the model to predict speech or voicing.

import argparse
import lightning as L
import yaml
import torch

import constants

from data.dataset import ProbingDataset, PaddingCollator
from models.prober import Prober
from models.pretrainer import Pretrainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

torch.set_float32_matmul_precision('high')

argparser = argparse.ArgumentParser(description="Pretrain a model")
argparser.add_argument(
    "--datasets_config",
    type=str,
    default="./datasets.yaml",
    help="Path to the datasets config file",
)

argparser.add_argument(
    "--training_config",
    type=str,
    default="./config.yaml",
    help="Path to the training config file",
)

argparser.add_argument(
    "--datasets",
    type=str,
    default=["armeni2022"],
    nargs="+",
    help="List of datasets to use for training (armeni2022, gwilliams2022)",
)

argparser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)

argparser.add_argument(
    "--name",
    type=str,
    default="probing",
    help="Name of the experiment",
)

argparser.add_argument(
    "--task",
    type=str,
    default="speech",
    help="Probing task (speech, voicing)",
)

argparser.add_argument(
    "--pretrained_ckpt",
    type=str,
    help="Path to the pretrained model checkpoint",
    required=True,
)

argparser.add_argument(
    "--debug",
    action="store_true",
    help="Run in debug mode (overfit batches, limit train batches, log every n steps)",
)

args = argparser.parse_args()

L.seed_everything(args.seed)

# Load the training config
with open(args.training_config, "r") as f:
    training_config = yaml.safe_load(f)


# Load the datasets config
with open(args.datasets_config, "r") as f:
    datasets_config = yaml.safe_load(f)

# Load pre-trained backbone
pretrainer = Pretrainer.load_from_checkpoint(
    args.pretrained_ckpt,
    strict=False,
)
backbone = pretrainer.backbone
n_pretrained_datasets = len(pretrainer.datasets)

print(f"Found backbone trained with {n_pretrained_datasets} datasets ({pretrainer.datasets}).")
print("Incrementing dataset ids by this much in probing.")

# Load datasets

train_sets, val_sets, test_sets = [], [], []
for i, dataset_name in enumerate(args.datasets):
    if dataset_name not in datasets_config:
        raise ValueError(f"Dataset {dataset_name} not found in config file.")
        
    train_sets.append(ProbingDataset(
        dataset_name=dataset_name,
        datasets_config=datasets_config,
        split="train",
        dataset_id=i + n_pretrained_datasets,
        data_task=args.task,
    ))

    val_sets.append(ProbingDataset(
        dataset_name=dataset_name,
        datasets_config=datasets_config,
        split="val",
        dataset_id=i + n_pretrained_datasets,
        data_task=args.task,
    ))

    test_sets.append(ProbingDataset(
        dataset_name=dataset_name,
        datasets_config=datasets_config,
        split="test",
        dataset_id=i + n_pretrained_datasets,
        data_task=args.task,
    ))

train_sets = torch.utils.data.ConcatDataset(train_sets)
val_sets = torch.utils.data.ConcatDataset(val_sets)
test_sets = torch.utils.data.ConcatDataset(test_sets)

collator = PaddingCollator(
    max_pad=constants.MAX_PAD,
)

train_loader = torch.utils.data.DataLoader(
    train_sets,
    batch_size=training_config["batch_size"],
    shuffle=True,
    num_workers=training_config["num_workers"],
    pin_memory=True,
    collate_fn=collator
)

val_loader = torch.utils.data.DataLoader(
    val_sets,
    batch_size=training_config["batch_size"],
    shuffle=False,
    num_workers=training_config["num_workers"],
    pin_memory=True,
    collate_fn=collator
)

test_loader = torch.utils.data.DataLoader(
    test_sets,
    batch_size=training_config["batch_size"],
    shuffle=False,
    num_workers=training_config["num_workers"],
    pin_memory=True,
    collate_fn=collator
)

# Define the model
prober = Prober(
    backbone=backbone,
    model_dim=training_config["model_dim"],
    subject_embedding_dim=training_config["subject_embedding_dim"],
    learning_rate=training_config["probe_learning_rate"],
)

checkpoint = ModelCheckpoint(
    monitor="val_auroc",
    dirpath="checkpoints",
    filename=args.name + "-best-{epoch:02d}-{val_auroc:.2f}",
    save_top_k=1,
    mode="max",
)

early_stopping = EarlyStopping(
    monitor="val_auroc",
    patience=10,
    mode="max",
)

logger = WandbLogger(
    name=args.name,
    project="the-brains-bitter-lesson",
    log_model=False,
)

trainer_params = dict(
    callbacks = [checkpoint, early_stopping],
    logger = logger,
    accelerator = "auto",
    devices = 1,
    max_epochs = training_config["max_epochs"],
)

if args.debug:
    trainer_params.update(
        overfit_batches=1,
        limit_train_batches=1,
        log_every_n_steps=1,
    )

trainer = L.Trainer(
    **trainer_params,
)

trainer.fit(
    prober,
    train_loader,
    val_loader,
)
prober = prober.load_from_checkpoint(
    checkpoint.best_model_path,
)
trainer.test(prober, test_loader)