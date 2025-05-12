import math
import random
import torch

class PhasePretext(torch.nn.Module):
    def __init__(self, input_dim, num_steps=8, prop=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.num_steps = num_steps
        self.prop = prop

        self.classifier = torch.nn.Linear(
            in_features=input_dim,
            out_features=self.num_steps,
        )
    
    def phase_shift(self, x):  # Assume x is [B, C, T]
        _, C, _ = x.shape

        # Randomly determine the phase shift
        possible_shifts = torch.linspace(
            start=0, end=math.pi - 0.125 * math.pi, steps=8, device=x.device
        )
        phase_shift_label = random.randrange(len(possible_shifts))
        phase_shift = possible_shifts[phase_shift_label]

        # FFT to convert the signal to frequency domain, using full FFT
        freq_x = torch.fft.fft(x, dim=2)

        # Calculate the phase shift factor in the complex plane
        phase_shift_factor = torch.exp(phase_shift * 1j)

        # Randomly select a proportion of the channels to apply the phase shift
        channels_to_shift = torch.randperm(C)[: int(C * self.prop)]

        # Apply the phase shift to the randomly selected channels
        freq_x[:, channels_to_shift, :] *= phase_shift_factor

        # Inverse FFT to convert back to time domain, using full IFFT
        time_x = torch.fft.ifft(
            freq_x, dim=2
        ).real  # Taking the real part since the original signal is real

        return time_x, phase_shift_label
    
    def forward(self, shifted_x, shift_label):
        z = self.classifier(shifted_x)

        label = torch.full(
            size=(shifted_x.shape[0],),
            fill_value=shift_label,
            dtype=torch.long,
            device=shifted_x.device,
        )

        loss = torch.nn.functional.cross_entropy(z, label)
        probs = torch.nn.functional.softmax(z, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        return loss, preds, label