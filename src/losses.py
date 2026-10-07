import torch
import torch.nn.functional as F


def contrastive_loss(
    z1,
    z2,
    label,
    margin=1.0
):
    """
    Contrastive loss for Siamese metric learning.

    Parameters
    ----------
    z1:
        Embedding of the first signal.

    z2:
        Embedding of the second signal.

    label:
        1 if the two signals belong to the same class.
        0 if they belong to different classes.

    margin:
        Minimum desired distance between different classes.
    """

    distance = F.pairwise_distance(z1, z2)

    positive_loss = (
        label * distance.pow(2)
    )

    negative_loss = (
        (1 - label)
        * torch.clamp(
            margin - distance,
            min=0.0
        ).pow(2)
    )

    return (
        positive_loss + negative_loss
    ).mean()