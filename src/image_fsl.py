from pathlib import Path

import numpy as np
import torch

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.evaluation import evaluate_predictions
from src.fsl import create_episode


def load_image(
    image_root,
    global_index,
    labels,
    class_names,
    representation=None
):

    class_id = labels[global_index]

    class_name = class_names[
        class_id
    ]

    local_index = global_index % 100

    image_root = Path(image_root)

    if representation is not None:
        image_root = image_root / representation

    path = image_root / class_name / f"{class_name}_{local_index:03d}.png"

    image = Image.open(
        path
    ).convert("L")

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0)

    return tensor


class ImageSignalDataset(Dataset):

    def __init__(
        self,
        image_root,
        indices,
        labels,
        class_names,
        representation=None
    ):
        self.image_root = image_root
        self.indices = np.asarray(indices)
        self.labels = np.asarray(labels)
        self.class_names = class_names
        self.representation = representation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        global_index = int(self.indices[position])
        image = load_image(
            self.image_root,
            global_index,
            self.labels,
            self.class_names,
            self.representation
        )
        return image, global_index


def get_image_embeddings(
    model,
    image_root,
    indices,
    labels,
    class_names,
    device,
    batch_size=32,
    representation=None
):
    """Extract embeddings for selected individual images."""

    dataset = ImageSignalDataset(
        image_root,
        indices,
        labels,
        class_names,
        representation
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model.eval()
    embeddings = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            batch_embeddings = model.encoder(images)
            embeddings.append(
                batch_embeddings.cpu().numpy()
            )

    if not embeddings:
        raise ValueError("Cannot embed an empty image selection")

    result = np.concatenate(embeddings, axis=0)

    if not np.isfinite(result).all():
        raise ValueError("Image embeddings contain non-finite values")

    return result


def compute_image_prototypes(
    support_embeddings,
    support_labels
):
    """Compute one prototype per episode class."""

    classes = np.unique(support_labels)
    prototypes = []

    for class_id in classes:
        class_embeddings = support_embeddings[
            support_labels == class_id
        ]
        prototypes.append(class_embeddings.mean(axis=0))

    return np.stack(prototypes)


def classify_image_queries(
    query_embeddings,
    prototypes
):
    distances = np.linalg.norm(
        query_embeddings[:, None, :]
        - prototypes[None, :, :],
        axis=2
    )
    predictions = np.argmin(distances, axis=1)
    return predictions, distances


def evaluate_image_fsl(
    model,
    image_root,
    test_indices,
    labels,
    class_names,
    device,
    k_shot,
    n_episodes=100,
    query_per_class=5,
    seed=42
):
    """Evaluate image embeddings with repeated few-shot episodes."""

    rng = np.random.default_rng(seed)
    all_true = []
    all_pred = []
    all_distances = []

    test_indices = np.asarray(test_indices)
    test_embeddings = get_image_embeddings(
        model,
        image_root,
        test_indices,
        labels,
        class_names,
        device
    )
    embedding_positions = {
        int(global_index): position
        for position, global_index in enumerate(test_indices)
    }

    for _ in range(n_episodes):
        episode = create_episode(
            indices=np.asarray(test_indices),
            labels=np.asarray(labels),
            n_way=len(class_names),
            k_shot=k_shot,
            query_per_class=query_per_class,
            rng=rng
        )

        support_positions = [
            embedding_positions[int(index)]
            for index in episode.support_indices
        ]
        query_positions = [
            embedding_positions[int(index)]
            for index in episode.query_indices
        ]

        support_embeddings = test_embeddings[support_positions]
        query_embeddings = test_embeddings[query_positions]

        prototypes = compute_image_prototypes(
            support_embeddings,
            episode.support_labels
        )
        predictions, distances = classify_image_queries(
            query_embeddings,
            prototypes
        )

        all_true.extend(episode.query_labels)
        all_pred.extend(predictions)
        all_distances.extend(distances)

    all_true = np.asarray(all_true)
    all_pred = np.asarray(all_pred)
    all_distances = np.asarray(all_distances)

    if set(np.unique(all_true)) != set(range(len(class_names))):
        raise ValueError("Aggregated queries do not contain every class")

    if not np.isfinite(all_distances).all():
        raise ValueError("Query distances contain non-finite values")

    results = evaluate_predictions(
        all_true,
        all_pred,
        all_distances
    )
    results["y_true"] = all_true
    results["y_pred"] = all_pred
    results["distances"] = all_distances
    return results