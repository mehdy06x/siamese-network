from pathlib import Path

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset


class SiameseImagePairDataset(
    Dataset
):

    def __init__(
        self,
        image_root,
        indices,
        labels,
        class_names,
        pairs_per_signal=4
    ):

        self.image_root = Path(
            image_root
        )

        self.indices = np.asarray(
            indices
        )

        self.labels = np.asarray(
            labels
        )

        self.class_names = class_names

        self.pairs = []

        rng = np.random.default_rng(42)

        for idx in self.indices:

            current_class = self.labels[idx]

            same = self.indices[
                self.labels[self.indices]
                == current_class
            ]

            different = self.indices[
                self.labels[self.indices]
                != current_class
            ]

            for _ in range(
                pairs_per_signal
            ):

                other = rng.choice(same)

                self.pairs.append(
                    (idx, other, 1.0)
                )

                other = rng.choice(
                    different
                )

                self.pairs.append(
                    (idx, other, 0.0)
                )

    def _load_image(self, idx):

        class_id = self.labels[idx]

        class_name = self.class_names[
            class_id
        ]

        file_name = (
            f"{class_name}_{idx % 100:03d}.png"
        )

        path = (
            self.image_root
            / class_name
            / file_name
        )

        image = Image.open(
            path
        ).convert("L")

        image = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        tensor = torch.from_numpy(
            image
        )

        tensor = tensor.unsqueeze(0)

        return tensor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        idx1, idx2, label = (
            self.pairs[index]
        )

        x1 = self._load_image(idx1)
        x2 = self._load_image(idx2)

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return x1, x2, label