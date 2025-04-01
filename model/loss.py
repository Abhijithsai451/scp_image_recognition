import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy(output, target):
    return F.cross_entropy(output, target)

def weighted_cross_entropy(output, target):
    class_counts = torch.bincount(target)
    class_weights = 1.0 / (class_counts.float() + 1e-6)  # Avoid division by zero
    class_weights /= class_weights.sum()  # Normalize weights

    # Move weights to the same device as output
    class_weights = class_weights.to(output.device)

    # Compute loss with weights
    loss = F.cross_entropy(output, target, weight=class_weights)

    return F.cross_entropy(output, target, weight=class_weights)