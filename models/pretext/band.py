import torch
import torchaudio
import random

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

        sfreq = sfreq[0].item()
        rejected_band_idx = random.randrange(len(self.bands))

        low_cutoff, high_cutoff = self.bands[rejected_band_idx]

        # Create a bandpass filter
        central_freq = (low_cutoff + high_cutoff) / 2
        bandwidth = high_cutoff - low_cutoff
        q_factor = central_freq / bandwidth

        filtered_x = torchaudio.functional.bandreject_biquad(
            x,
            sample_rate=sfreq,
            central_freq=central_freq,
            Q=q_factor,
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

        return loss, preds, label