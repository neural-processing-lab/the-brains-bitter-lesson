# Pretrain a model

import argparse
import lightning as L
import yaml
import torch

from data.dataset import PretrainingDataset, PaddingCollator
from models.pretrainer import Pretrainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

# Must be larger than the largest sensor dim. in the data
MAX_PAD = 306

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
    default="armeni2022",
    nargs="+",
    help="List of datasets to use for training (camcan, mous, armeni2022, gwilliams2022)",
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
    default="pretraining",
    help="Name of the experiment",
)

argparser.add_argument(
    "--debug",
    action="store_true",
    help="Run in debug mode (overfit batches, limit train batches, log every n steps)",
)

args = argparser.parse_args()

L.seed_everything(args.seed)

# Load the training config
with open(argparser.training_config, "r") as f:
    training_config = yaml.safe_load(f)


# Load the datasets config
with open(argparser.datasets_config, "r") as f:
    datasets_config = yaml.safe_load(f)

# Load datasets
train_sets, val_sets, test_sets = [], [], []
for dataset_name in args.datasets:
    if dataset_name not in datasets_config:
        raise ValueError(f"Dataset {dataset_name} not found in config file.")
    
    train_sets.append(PretrainingDataset(
        dataset_name=dataset_name,
        datasets_config=datasets_config,
        split="train",
    ))

    val_sets.append(PretrainingDataset(
        dataset_name=dataset_name,
        datasets_config=datasets_config,
        split="val",
    ))

    test_sets.append(PretrainingDataset(
        dataset_name=dataset_name,
        datasets_config=datasets_config,
        split="test",
    ))

train_sets = torch.utils.data.ConcatDataset(train_sets)
val_sets = torch.utils.data.ConcatDataset(val_sets)
test_sets = torch.utils.data.ConcatDataset(test_sets)

collator = PaddingCollator(
    max_pad=MAX_PAD,
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
pretrainer = Pretrainer(
    input_dim=MAX_PAD,
    model_dim=training_config["model_dim"],
    subject_embedding_dim=training_config["subject_embedding_dim"],
    flat_dim=training_config["flat_dim"],
    projector_dim=training_config["projector_dim"],
    learning_rate=training_config["learning_rate"],
)

checkpoint = ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints",
    filename=args.name + "-best-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,
    mode="min",
)

logger = WandbLogger(
    name=args.name,
    project="the-brains-bitter-lesson",
    log_model=False,
)

trainer_params = dict(
    callbacks = [checkpoint],
    logger = logger,
    acclerator = "auto",
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
    pretrainer,
    train_loader,
    val_loader,
)
pretrainer = pretrainer.load_from_checkpoint(
    checkpoint.best_model_path,
)
trainer.test(pretrainer, test_loader)