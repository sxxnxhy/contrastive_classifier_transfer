
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import config
from dataset import prepare_dataloader
from model_transfer import TransferModel


def main(num_samples=config.NUM_SAMPLES, seed=config.SEED):
    device = config.DEVICE
    print(f"Device: {device}")

    # Get loaders (use small number of workers to be safe)
    _, val_loader, classes = prepare_dataloader(
        use_lazy_loading=config.USE_LAZY_LOADING,
        use_parallel=False,
        max_workers=config.MAX_WORKERS,
        num_workers=config.NUM_WORKERS,
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

    # Deterministically sample `num_samples` indices from the validation dataset
    val_dataset = val_loader.dataset
    dataset_len = len(val_dataset)
    if num_samples > dataset_len:
        raise ValueError(f"Requested {num_samples} samples but validation dataset has {dataset_len}")

    rng = np.random.RandomState(seed)
    sample_indices = rng.choice(dataset_len, size=num_samples, replace=False)

    spectra_list = []
    labels_list = []
    for idx in sample_indices:
        item = val_dataset[idx]
        # item is (spectrum, labels_and_mod)
        spectra_list.append(item[0])
        labels_list.append(item[1])

    if len(spectra_list) == 0:
        print("No validation samples found.")
        return

    batch_spectra = torch.stack(spectra_list, dim=0).to(device)
    labels_and_mod = torch.stack(labels_list, dim=0).to(device)
    labels = labels_and_mod[:, :-1]
    modalities = labels_and_mod[:, -1]

    with torch.no_grad():
        # model(...) returns embeddings; run classifier to get logits
        embeddings = model(batch_spectra, modalities)
        logits = model.classifier(embeddings)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        targets = torch.argmax(labels, dim=1)

    # Diagnostics: check logits/classes shape
    num_logits = logits.size(1)
    if num_logits != num_classes:
        print(f"WARNING: model output dim ({num_logits}) != number of classes ({num_classes}).\n" \
              "This indicates a mismatch between the saved checkpoint and current class list.")

    correct = (preds == targets).sum().item()
    acc = correct / preds.size(0)

    print(f"Test samples: {preds.size(0)}  Accuracy: {acc:.3f}")

    for i in range(preds.size(0)):
        t_idx = targets[i].item()
        p_idx = preds[i].item()

        cls_true = classes[t_idx] if 0 <= t_idx < num_classes else f"IDX_{t_idx}"
        cls_pred = classes[p_idx] if 0 <= p_idx < num_classes else f"IDX_{p_idx}"
        # safe prob extraction
        prob = probs[i, p_idx].item() if 0 <= p_idx < probs.size(1) else float('nan')
        # modality (0.0=raman, 1.0=gc)
        mod_val = modalities[i].item()
        mod_str = 'raman' if float(mod_val) < 0.5 else 'gc'
        print(f"{i:02d}: mod={mod_str:5s} true={cls_true:12s} pred={cls_pred:12s} p={prob:.3f}")


if __name__ == '__main__':
    main()
