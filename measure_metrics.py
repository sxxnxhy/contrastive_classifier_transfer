"""Measure metrics for a trained model checkpoint.

Usage:
    python measure_metrics.py --checkpoint models/ResNet.pth

This script will:
- load the checkpoint (robust to several formats)
- load the validation DataLoader via dataset.prepare_dataloader
- run the encoder (and classifier if present) on the val set
- compute classification metrics (accuracy, precision, recall, f1, f_beta, auc)
- compute k-NN evaluation on embeddings for a complementary metric
- print and save results to a JSON file next to the checkpoint
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import config
from dataset import prepare_dataloader
from model_transfer import TransferModel
from metrics import classification_metrics, knn_evaluate


def load_checkpoint(path: str, device='cpu'):
    ckpt = torch.load(path, map_location=device)

    # find model_state-like entries
    possible_keys = ['model_state', 'model_state_dict', 'state_dict', 'model_state_dicts']
    model_state = None
    for k in possible_keys:
        if k in ckpt:
            model_state = ckpt[k]
            break

    if model_state is None:
        model_state = ckpt

    return ckpt, model_state





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='models/transfer_contrastive_classifier.pth')
    parser.add_argument('--use-lazy', action='store_true', default=config.USE_LAZY_LOADING,
                        help='Use lazy loading for dataset')
    parser.add_argument('--num-workers', type=int, default=config.NUM_WORKERS)
    parser.add_argument('--knn-k', type=int, default=5)
    parser.add_argument('--save', action='store_true', help='Save JSON results next to checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Prepare dataloaders (we only need val_loader)
    train_loader, val_loader, classes = prepare_dataloader(
        use_lazy_loading=args.use_lazy,
        use_parallel=False,
        max_workers=1,
        num_workers=args.num_workers
    )

    # Build transfer model (encoder + classifier)
    num_classes = len(classes)
    model = TransferModel(hidden_dim=config.HIDDEN_DIM, embed_dim=config.EMBEDDING_DIM, num_classes=num_classes)
    model.to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt, model_state = load_checkpoint(str(ckpt_path), device=device)

    # Try loading into TransferModel (handles encoder + classifier keys). Use strict=False to be robust.
    has_classifier = False
    try:
        if isinstance(model_state, dict):
            try:
                model.load_state_dict(model_state, strict=False)
            except Exception:
                # try nested dicts
                loaded = False
                for v in model_state.values():
                    if isinstance(v, dict):
                        try:
                            model.load_state_dict(v, strict=False)
                            loaded = True
                            break
                        except Exception:
                            continue
                if not loaded:
                    print("Warning: could not load model weights cleanly; proceeding with init weights")
        else:
            try:
                model.load_state_dict(model_state)
            except Exception:
                print("Warning: checkpoint format unsupported for direct loading; proceeding with init weights")

        # Determine whether classifier weights are present in the loaded state
        if isinstance(model_state, dict):
            has_classifier = any(k.startswith('classifier.') or 'classifier' in k for k in model_state.keys())
        else:
            has_classifier = hasattr(model, 'classifier')
    except Exception as e:
        print(f"Warning while loading checkpoint: {e}")

    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    all_embeddings = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for spectra, labels_and_mod in tqdm(val_loader, desc='Evaluating'):
            spectra = spectra.to(device)

            labels_and_mod = labels_and_mod.to(device)

            # model.forward expects (spectra, modalities)
            embeddings = model(spectra, labels_and_mod[:, -1])
            all_embeddings.append(embeddings.cpu().numpy())

            # If classifier weights were loaded (or present), use classifier
            try:
                logits = model.classifier(embeddings)
                probs = softmax(logits)
                preds = torch.argmax(probs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
            except Exception:
                # classifier not usable
                pass

            all_labels.append(labels_and_mod.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
    else:
        all_preds = None
    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs, axis=0)
    else:
        all_probs = None

    results = {}

    if all_preds is not None:
        print("Computing classification metrics from classifier outputs...")
        cm = classification_metrics(all_labels, all_preds, y_score=all_probs, average='macro')
        results['classifier_metrics'] = cm

    # Always run k-NN on embeddings as a complementary evaluation
    print(f"Running k-NN (k={args.knn_k}) on {all_embeddings.shape[0]} embeddings...")
    knn_out = knn_evaluate(all_embeddings, all_labels, k=args.knn_k, test_size=0.2)
    results['knn'] = knn_out

    print(json.dumps(results, indent=2, default=lambda x: str(x)))

    if args.save:
        out_path = ckpt_path.with_suffix('.metrics.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: str(x))
        print(f"Saved results to {out_path}")


if __name__ == '__main__':
    main()
