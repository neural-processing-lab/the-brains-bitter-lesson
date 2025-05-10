# Defines the backbone of the model

import torch

from .multisensor_projector import MultisensorProjector
from .seanet.seanet import SEANetBrainEncoder

class Backbone(torch.nn.Module):
    def __init__(self, input_dim, model_dim, subject_embedding_dim):

        self.subject_embedding = torch.nn.Embedding(
            num_embeddings=128, # Allows for up to this many subjects. NOTE: if averaging, only use trained embeddings
            embedding_dim=subject_embedding_dim,
        )

        # NOTE: Dataset IDs need to be incremented when datasets are pooled
        self.dataset_gating = MultisensorProjector(
            sensor_dim=input_dim,
            model_dim=model_dim,
            num_datasets=4, # NOTE: set this to max number of datasets
        )

        self.encoder = SEANetBrainEncoder(
            channels=model_dim,
        )

        self.trained_subjects = set()
    
    def forward(self, x, dataset_id, subject_id, use_mean_subject_embedding=False):

        # NOTE: inputs should already be padded to the same length

        x = self.dataset_gating(x, dataset_id=dataset_id)
        x = self.encoder(x) # [B, E, T]

        if not use_mean_subject_embedding:
            z_subject = self.subject_embedding(subject_id).unsqueeze(-1) # [B, S, 1]
            x = torch.cat([x, z_subject], dim=1) # [B, E + S, T]
            self.trained_subjects = self.trained_subjects.union(
                set(subject_id.squeeze().tolist())
            )
        else:
            # Use mean subject embedding for all trained subjects
            z_subject = self.subject_embedding.weight[:max(self.trained_subjects) + 1].mean(dim=0).unsqueeze(0).unsqueeze(-1) # [1, S, 1]
            x = torch.cat([x, z_subject], dim=1)

        return x

