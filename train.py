"""Transfer-training script: freeze pretrained encoders and train a classifier head.

This script:
- Loads the contrastive model and weights from `config.CHECKPOINT_PATH` (if present)
- Freezes all existing model parameters
- Attaches a new linear classifier on top of the embedding
- Trains only the classifier parameters using the dataset returned by
  `dataset.prepare_dataloader`
- Saves the final state_dict to `config.SAVE_DIR`

No argparse is used — edit `config.py` for hyperparameters.
"""
from pathlib import Path
import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from dataset import prepare_dataloader
from model import CrossModalContrastiveModel


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_pretrained_weights(model: torch.nn.Module, path: str, device: str):
    if not Path(path).exists():
        print(f"Checkpoint not found at {path}. Proceeding with random init.")
        return

    print(f"Loading checkpoint from {path} ...")
    ckpt = torch.load(path, map_location=device)

    # ckpt might be a state_dict or a dict with 'model_state_dict'
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and any(k.startswith('module.') or k in model.state_dict() for k in ckpt.keys()):
        state = ckpt
    else:
        state = ckpt

    model.load_state_dict(state, strict=False)
    print("Checkpoint loaded (strict=False). Some keys may be unmatched — that's expected for transfer learning.")


def main():
    device = config.DEVICE
    print(f"Using device: {device}")

    # Prepare data loaders
    print("Preparing dataloaders...")
    train_loader, val_loader, classes = prepare_dataloader(
        use_lazy_loading=config.USE_LAZY_LOADING,
        use_parallel=True,
        max_workers=4,
        num_workers=config.NUM_WORKERS,
    )

    num_classes = len(classes)
    print(f"Found {num_classes} classes: {classes}")

    # Build model and load pretrained weights
    model = CrossModalContrastiveModel(hidden_dim=config.HIDDEN_DIM, embed_dim=config.EMBEDDING_DIM)
    load_pretrained_weights(model, config.CHECKPOINT_PATH, device)

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    # Attach classifier head
    classifier = nn.Linear(model.embed_dim, num_classes)
    # He init
    nn.init.kaiming_normal_(classifier.weight, nonlinearity='linear')
    if classifier.bias is not None:
        nn.init.constant_(classifier.bias, 0.0)

    # Attach to model so it's saved together
    model.classifier = classifier

    model.to(device)

    # Only classifier params will be trainable
    for p in model.classifier.parameters():
        p.requires_grad = True

    print(f"Trainable parameters: {count_trainable_params(model)}")

    # Optimizer / loss
    optimizer = optim.Adam(model.classifier.parameters(), lr=config.LR)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_samples = 0
        train_correct = 0

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False)
        for batch_idx, batch in enumerate(train_iter):
            spectra, labels_and_mod = batch
            spectra = spectra.to(device)
            labels_and_mod = labels_and_mod.to(device)

            labels = labels_and_mod[:, :-1]
            modalities = labels_and_mod[:, -1]

            # Convert one-hot labels to class indices for CrossEntropyLoss
            targets = torch.argmax(labels, dim=1).long()

            optimizer.zero_grad()
            embeddings = model(spectra, modalities)
            logits = model.classifier(embeddings)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            bs = targets.shape[0]
            running_loss += loss.item() * bs
            n_samples += bs

            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == targets).sum().item()

            batch_acc = (preds == targets).float().mean().item()

            # update tqdm postfix with loss, lr and batch acc
            train_iter.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}", 'acc': f"{batch_acc:.3f}"})

        epoch_loss = running_loss / max(1, n_samples)
        epoch_acc = train_correct / max(1, n_samples)
        t1 = time.time()
        print(f"Epoch {epoch} train loss: {epoch_loss:.4f}  ACC: {epoch_acc:.4f}  time: {t1 - t0:.1f}s")

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        correct = 0
        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch} val", leave=False)
            for spectra, labels_and_mod in val_iter:
                spectra = spectra.to(device)
                labels_and_mod = labels_and_mod.to(device)

                labels = labels_and_mod[:, :-1]
                modalities = labels_and_mod[:, -1]

                targets = torch.argmax(labels, dim=1).long()

                embeddings = model(spectra, modalities)
                logits = model.classifier(embeddings)
                loss = criterion(logits, targets)

                bs = targets.shape[0]
                val_loss += loss.item() * bs
                val_samples += bs

                # Compute simple accuracy using argmax (single-label assumption)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == targets).sum().item()

                # update val tqdm postfix with running averages
                avg_val_loss = val_loss / max(1, val_samples)
                avg_val_acc = correct / max(1, val_samples)
                val_iter.set_postfix({'val_loss': f"{avg_val_loss:.4f}", 'val_acc': f"{avg_val_acc:.4f}"})

        val_loss = val_loss / max(1, val_samples)
        val_acc = correct / max(1, val_samples)
        print(f"Epoch {epoch} VAL loss: {val_loss:.4f}  ACC: {val_acc:.4f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = Path(config.SAVE_DIR)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(save_path))
            print(f"Saved best model to {save_path} (val_loss={val_loss:.4f})")

    print("Training finished.")


if __name__ == '__main__':
    main()
