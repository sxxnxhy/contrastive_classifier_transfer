
from pathlib import Path
import torch
import torch.nn.functional as F

import config
from dataset import prepare_dataloader
from model_transfer import TransferModel


def main(num_samples: int = 10):
    device = config.DEVICE
    print(f"Device: {device}")

    # Get loaders (use small number of workers to be safe)
    _, val_loader, classes = prepare_dataloader(
        use_lazy_loading=config.USE_LAZY_LOADING,
        use_parallel=False,
        max_workers=1,
        num_workers=0,
    )

    num_classes = len(classes)
    print(f"Classes: {classes}")

    # Build transfer model and load weights
    model = TransferModel(hidden_dim=config.HIDDEN_DIM, embed_dim=config.EMBEDDING_DIM, num_classes=num_classes)
    ckpt_path = Path(config.SAVE_DIR)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Saved model not found at {ckpt_path}. Run training first.")

    print(f"Loading saved model from {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location=device)
    # Allow partial matches
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # Collect up to num_samples from val set
    spectra_list = []
    labels_list = []
    modalities_list = []
    collected = 0
    for batch in val_loader:
        spectra, labels_and_mod = batch
        b = spectra.shape[0]
        for i in range(b):
            spectra_list.append(spectra[i])
            labels_list.append(labels_and_mod[i])
            collected += 1
            if collected >= num_samples:
                break
        if collected >= num_samples:
            break

    if len(spectra_list) == 0:
        print("No validation samples found.")
        return

    batch_spectra = torch.stack(spectra_list, dim=0).to(device)
    labels_and_mod = torch.stack(labels_list, dim=0).to(device)
    labels = labels_and_mod[:, :-1]
    modalities = labels_and_mod[:, -1]

    with torch.no_grad():
        logits = model(batch_spectra, modalities)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        targets = torch.argmax(labels, dim=1)

    correct = (preds == targets).sum().item()
    acc = correct / preds.size(0)

    print(f"Test samples: {preds.size(0)}  Accuracy: {acc:.3f}")

    for i in range(preds.size(0)):
        cls_true = classes[targets[i].item()]
        cls_pred = classes[preds[i].item()]
        prob = probs[i, preds[i]].item()
        print(f"{i:02d}: true={cls_true:12s} pred={cls_pred:12s} p={prob:.3f}")


if __name__ == '__main__':
    main(num_samples=10)
