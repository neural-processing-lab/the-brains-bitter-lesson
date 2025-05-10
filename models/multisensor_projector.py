import torch

class MultisensorProjector(torch.nn.Module):
    """A convolutional projector.

    This module takes tensors from multiple datasets (padded to the same sensor dimension) and projects them to a
    shared space. It uses dataset-conditional adapters to harmonize the data from different datasets in latent space.
    """

    def __init__(self, sensor_dim, model_dim, num_datasets=None):
        super().__init__()

        self.num_datasets = num_datasets
        self.use_dataset_embedding = num_datasets > 1

        input_dim = sensor_dim

        if self.use_dataset_embedding:
            self.projector = self.build_multisensor_projector(input_dim, num_datasets * model_dim)
        else:
            self.projector = self.build_multisensor_projector(input_dim, model_dim)

    def build_multisensor_projector(self, sensor_dim, model_dim):
        conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(sensor_dim, model_dim, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv1d(model_dim, model_dim, kernel_size=3, stride=2, padding=1),
        )
        conv1.stride = (2,)
        return conv1

    def forward(self, x, dataset_id=None):

        if self.use_dataset_embedding:
            # Expect data to already come padded in sensor dimension to uniform size.
            self.dataset_onehot = torch.nn.functional.one_hot(
                dataset_id,
                num_classes=self.num_datasets
            ) # [B, M]

            # Project (using dataset-conditional weights in conv)
            x = self.projector(x) # [B, M * C, T]

            # Apply gating using one-hot dataset embedding
            x = torch.unflatten(x, 1, (self.num_datasets, -1)) # [B, M, C, T]
            x = (x * self.dataset_onehot[:, :, None, None]).sum(dim=1) # [B, C, T]
        else:
            x = self.projector(x)

        return x