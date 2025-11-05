"""Transfer-ready model architecture for inference.

This module provides a TransferModel that matches the `model` object
saved by `train.py` (i.e., a CrossModalContrastiveModel with a
`.classifier` Linear head). Use this to load the saved state_dict and
perform inference.
"""
import torch
import torch.nn as nn
from model import CrossModalContrastiveModel


class TransferModel(CrossModalContrastiveModel):
    """A convenience subclass that attaches a classifier head with the
    same attribute name used when saving in `train.py`.

    Example:
        model = TransferModel(hidden_dim=..., embed_dim=..., num_classes=4)
        model.load_state_dict(torch.load(PATH))
        model.eval()
        logits = model(spectra, modalities)
"""
    def __init__(self, hidden_dim: int, embed_dim: int, num_classes: int):
        super().__init__(hidden_dim=hidden_dim, embed_dim=embed_dim)
        # Attach classifier to match saved keys
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def predict(self, spectra: torch.Tensor, modalities: torch.Tensor) -> torch.Tensor:
        """Return class probabilities (softmax) for given batch.

        Args:
            spectra: (B, L) or (B, C, L)
            modalities: (B,) modality indicators

        Returns:
            probs: (B, num_classes) probabilities
        """
        if spectra.dim() == 2:
            # (B, L) → (B, L) expected by model.forward which handles unsqueezing
            pass

        with torch.no_grad():
            embeddings = self(spectra, modalities)
            logits = self.classifier(embeddings)
            probs = torch.softmax(logits, dim=1)
        return probs


__all__ = ["TransferModel"]
