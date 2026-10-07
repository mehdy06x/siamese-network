import numpy as np
import torch


class FewShotEpisode:

    def __init__(
        self,
        support_indices,
        query_indices,
        support_labels,
        query_labels
    ):
        self.support_indices = support_indices
        self.query_indices = query_indices
        self.support_labels = support_labels
        self.query_labels = query_labels


def get_embeddings(
    model,
    X_data,
    indices=None,
    device=None,
    batch_size=16
):
    """Generate encoder embeddings for selected 1D signals."""

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    data = np.asarray(X_data)

    if indices is not None:
        data = data[np.asarray(indices)]

    model.eval()
    embeddings = []

    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = torch.from_numpy(
                data[start:start + batch_size]
            ).float().unsqueeze(1).to(device)

            embeddings.append(
                model.encoder(batch).cpu().numpy()
            )

    return np.concatenate(embeddings, axis=0)


def create_episode(
    indices,
    labels,
    n_way=3,
    k_shot=1,
    query_per_class=5,
    rng=None
):
    """
    Create one N-way K-shot episode.

    Parameters
    ----------
    indices:
        Available signal indices.

    labels:
        Class label for every signal.

    n_way:
        Number of classes in the episode.

    k_shot:
        Number of support examples per class.

    query_per_class:
        Number of query examples per class.
    """

    if rng is None:
        rng = np.random.default_rng()

    classes = np.unique(labels)

    selected_classes = rng.choice(
        classes,
        size=n_way,
        replace=False
    )

    selected_classes = np.sort(
        selected_classes
    )

    support_indices = []
    query_indices = []
    support_labels = []
    query_labels = []

    for episode_label, class_id in enumerate(selected_classes):

        class_indices = indices[
            labels[indices] == class_id
        ]

        required = k_shot + query_per_class

        if len(class_indices) < required:
            raise ValueError(
                f"Not enough samples for class {class_id}"
            )

        chosen = rng.choice(
            class_indices,
            size=required,
            replace=False
        )

        support = chosen[:k_shot]
        query = chosen[k_shot:]

        support_indices.extend(support)
        query_indices.extend(query)

        support_labels.extend(
            [episode_label] * len(support)
        )

        query_labels.extend(
            [episode_label] * len(query)
        )

    return FewShotEpisode(
        support_indices=np.array(support_indices),
        query_indices=np.array(query_indices),
        support_labels=np.array(support_labels),
        query_labels=np.array(query_labels)
    )


def compute_prototypes(
    support_embeddings,
    support_labels
):
    """Compute one mean embedding prototype for each class."""

    prototypes = []

    for class_id in np.unique(support_labels):
        class_embeddings = support_embeddings[
            support_labels == class_id
        ]
        prototypes.append(class_embeddings.mean(axis=0))

    return np.stack(prototypes)


def classify_queries(
    query_embeddings,
    prototypes
):
    """Classify queries by their Euclidean distance to each prototype."""

    distances = np.linalg.norm(
        query_embeddings[:, None, :]
        - prototypes[None, :, :],
        axis=2
    )

    predictions = np.argmin(distances, axis=1)

    return predictions, distances


def get_embeddings(
    model,
    X,
    indices,
    device
):
    """
    Extract embeddings for selected signals.
    """

    model.eval()

    embeddings = []

    with torch.no_grad():

        for idx in indices:

            signal = torch.from_numpy(
                X[idx]
            ).float()

            signal = signal.unsqueeze(0)
            signal = signal.unsqueeze(0)

            signal = signal.to(device)

            embedding = model.encoder(
                signal
            )

            embeddings.append(
                embedding.squeeze(0)
                .cpu()
                .numpy()
            )

    return np.stack(embeddings)
def get_image_embeddings(
    model,
    image_loader,
    device
):

    model.eval()

    embeddings = []
    labels = []

    with torch.no_grad():

        for images, batch_labels in image_loader:

            images = images.to(device)

            z = model.encoder(
                images
            )

            embeddings.append(
                z.cpu().numpy()
            )

            labels.extend(
                batch_labels.numpy()
            )

    return (
        np.concatenate(embeddings),
        np.asarray(labels)
    )
