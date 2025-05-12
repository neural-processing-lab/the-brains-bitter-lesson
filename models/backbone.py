# Defines the backbone of the model

import torch

import constants

from .multisensor_projector import MultisensorProjector
from .seanet.seanet import SEANetBrainEncoder

class Backbone(torch.nn.Module):
    def __init__(self, input_dim, model_dim, subject_embedding_dim):
        super(Backbone, self).__init__()

        self.subject_embedding = torch.nn.Embedding(
            # Allow for so many subjects per dataset and up to 10 datasets
            num_embeddings=constants.MAX_SUBJECTS_PER_DATASET * constants.MAX_DATASETS,
            # NOTE: if averaging, only use trained embeddings
            embedding_dim=subject_embedding_dim,
        )

        # NOTE: Dataset IDs need to be incremented when datasets are pooled
        self.dataset_gating = MultisensorProjector(
            sensor_dim=input_dim,
            model_dim=model_dim,
            num_datasets=constants.MAX_DATASETS, # NOTE: set this to max number of datasets
        )

        self.encoder = SEANetBrainEncoder(
            channels=model_dim,
            dimension=model_dim,
        )

        self.trained_subjects = set()
    
    def forward(self, x, dataset_id, subject_id, use_mean_subject_embedding=False):

        # NOTE: inputs should already be padded to the same length

        x = self.dataset_gating(x, dataset_id=dataset_id)
        x = self.encoder(x) # [B, E, T]
        x = torch.mean(x, dim=-1) # [B, E]

        if not use_mean_subject_embedding:
            z_subject = self.subject_embedding(subject_id) # [B, S]
            x = torch.cat([x, z_subject], dim=1) # [B, E + S]
            self.trained_subjects = self.trained_subjects.union(
                set(subject_id.squeeze().tolist())
            )
        else:
            # Gather subject embeddings for all subjects in trained_subjects
            indices_tensor = torch.tensor(list(self.trained_subjects), dtype=torch.long, device=x.device)
            trained_embedddings = self.subject_embedding(indices_tensor) # [N, S]
            average_embedding = trained_embedddings.mean(dim=0).unsqueeze(0) # [1, S]
            x = torch.cat([x, average_embedding], dim=1) # [B, E + S]

        return x

