import torch
import torch.nn as nn

class RunningMeanStd(nn.Module):
    """
    Tracks the running mean and variance of a batch of data using Welford's algorithm.
    Normalizes observations and safely clips extreme outliers.
    """
    def __init__(self, shape: tuple, epsilon: float = 1e-8, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer('mean', torch.zeros(shape, dtype=torch.float32, device=device))
        self.register_buffer('var', torch.ones(shape, dtype=torch.float32, device=device))
        self.register_buffer('count', torch.tensor(1e-4, dtype=torch.float32, device=device))

    def update(self, x: torch.Tensor):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # Batched Welford's algorithm for updating mean and variance
        self.mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count

        self.var = M2 / total_count
        self.count = total_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize and clip to [-10, 10] to prevent catastrophic gradient explosions from physics glitches
        norm_x = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        return torch.clamp(norm_x, -10.0, 10.0)