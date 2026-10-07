import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DEncoder(nn.Module):

    def __init__(self, embedding_dim=128):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                1, 32,
                kernel_size=9,
                padding=4
            ),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(
                32, 64,
                kernel_size=7,
                padding=3
            ),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(
                64, 128,
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.embedding = nn.Linear(
            128,
            embedding_dim
        )

    def forward(self, x):

        x = self.features(x)
        x = x.flatten(1)

        return self.embedding(x)


class SiameseNetwork(nn.Module):

    def __init__(self, embedding_dim=128):

        super().__init__()

        # ONE encoder shared by both inputs
        self.encoder = Conv1DEncoder(
            embedding_dim=embedding_dim
        )

    def forward(self, x1, x2):

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        return z1, z2

class Conv2DEncoder(nn.Module):

    def __init__(self, embedding_dim=128):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1, 32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.embedding = nn.Linear(
            128,
            embedding_dim
        )

    def forward(self, x):

        x = self.features(x)

        x = x.flatten(1)

        return self.embedding(x)


class Siamese2D(nn.Module):

    def __init__(self, embedding_dim=128):

        super().__init__()

        self.encoder = Conv2DEncoder(
            embedding_dim
        )

    def forward(self, x1, x2):

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        return z1, z2