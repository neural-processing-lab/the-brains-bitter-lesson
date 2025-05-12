import random
import torch

class AmpPretext(torch.nn.Module):
    def __init__(self, input_dim, num_steps=16, prop=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.num_steps = num_steps
        self.prop = prop

        self.classifier = torch.nn.Linear(
            in_features=input_dim,
            out_features=self.num_steps,
        )
    
    def scale_amp(self, x):

        _, C, _ = x.shape

        possible_scales = torch.linspace(
            start=-2, end=2, steps=self.num_steps, device=x.device
        )
        scale_label = random.randrange(len(possible_scales))
        scale = possible_scales[scale_label]

        # Randomly select a proportion of the channels to apply the amplitude scaling to
        channels_to_scale = torch.randperm(C)[: int(C * self.prop)]

        x_scaled = x.clone()  # Avoids in-place gradient computation error
        x_scaled[:, channels_to_scale, :] *= scale

        return x_scaled, scale_label
    
    def forward(self, scaled_x, scale_label):
        z = self.classifier(scaled_x)

        label = torch.full(
            size=(scaled_x.shape[0],),
            fill_value=scale_label,
            dtype=torch.long,
            device=scaled_x.device,
        )

        loss = torch.nn.functional.cross_entropy(z, label)
        probs = torch.nn.functional.softmax(z, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        return loss, preds, label