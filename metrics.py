"""Common evaluation metrics for classification and embedding-based k-NN.

Provides:
- classification_metrics: accuracy, precision, recall, f1, f_beta, auc (if scores provided)
- knn_evaluate: train/test split k-NN classification on embeddings with metric report

Works with torch tensors or numpy arrays.
"""
from typing import Optional, Dict, Any

import numpy as np
import config
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier



def _to_numpy(x):
    if x is None:
        return None
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def classification_metrics(y_true, y_pred, y_score: Optional[np.ndarray] = None,
                           average: str = 'macro', beta: float = 1.0) -> Dict[str, Any]:
    """Compute common classification metrics.

    Args:
        y_true: (N,) ground-truth labels (ints or one-hot)
        y_pred: (N,) predicted labels
        y_score: (N,) or (N, C) probability or score for positive class(es) (optional, for AUC)
        average: averaging method for multi-class ('macro', 'micro', 'weighted')
        beta: beta for F-beta score (fvscore)

    Returns:
        dict with accuracy, precision, recall, f1, f_beta, auc (if score provided), conf_mat
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    y_score = _to_numpy(y_score)

    # If y_true is one-hot or contains modality appended (one-hot + modality), convert to indices
    n_classes = None
    try:
        n_classes = len(config.CLASSES)
    except Exception:
        n_classes = None

    def _strip_mod(colarr: np.ndarray) -> np.ndarray:
        # If 2D and last column is modality appended (so cols == n_classes+1), drop it
        if colarr.ndim == 2 and n_classes is not None and colarr.shape[1] == n_classes + 1:
            return colarr[:, :n_classes]
        return colarr

    y_true_proc = _strip_mod(y_true)
    y_pred_proc = _strip_mod(y_pred)

    if y_true_proc.ndim == 2:
        y_true_idx = np.argmax(y_true_proc, axis=1)
    else:
        y_true_idx = y_true_proc.astype(int)

    if y_pred_proc.ndim == 2:
        y_pred_idx = np.argmax(y_pred_proc, axis=1)
    else:
        y_pred_idx = y_pred_proc.astype(int)

    acc = float(accuracy_score(y_true_idx, y_pred_idx))
    prec = float(precision_score(y_true_idx, y_pred_idx, average=average, zero_division=0))
    rec = float(recall_score(y_true_idx, y_pred_idx, average=average, zero_division=0))
    f1 = float(f1_score(y_true_idx, y_pred_idx, average=average, zero_division=0))

    # F-beta (fvscore)
    if beta == 1.0:
        fbeta = f1
    else:
        # sklearn doesn't expose fbeta_score with average easily, so compute via f1_score variant
        from sklearn.metrics import fbeta_score
        fbeta = float(fbeta_score(y_true_idx, y_pred_idx, beta=beta, average=average, zero_division=0))

    auc = None
    if y_score is not None:
        try:
            # If binary and y_score is (N,), compute roc_auc_score directly
            if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 1):
                auc = float(roc_auc_score(y_true_idx, y_score.ravel()))
            else:
                # multiclass: require shape (N, C) and compute OvR macro AUC
                auc = float(roc_auc_score(y_true_idx, y_score, multi_class='ovr', average='macro'))
        except Exception:
            auc = None

    conf = confusion_matrix(y_true_idx, y_pred_idx)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'f_beta': fbeta,
        'auc': auc,
        'confusion_matrix': conf,
    }


def knn_evaluate(embeddings, labels, k: int = 5, test_size: float = 0.2,
                 random_state: int = 42, average: str = 'macro', n_jobs: int = 1) -> Dict[str, Any]:
    """Train a k-NN on `embeddings` and evaluate on a held-out test split.

    Args:
        embeddings: (N, D) array-like or tensor
        labels: (N,) labels (ints or one-hot)
        k: neighbours
        test_size: test split fraction
        random_state: RNG seed
        average: averaging for metrics
        n_jobs: parallel jobs for k-NN

    Returns:
        dict containing train/test sizes, metrics on test, and the fitted classifier
    """
    X = _to_numpy(embeddings)
    y = _to_numpy(labels)
    if y.ndim == 2:
        y = np.argmax(y, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)

    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=n_jobs)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    # If predict_proba available
    y_score = None
    if hasattr(knn, 'predict_proba'):
        y_score = knn.predict_proba(X_test)

    metrics = classification_metrics(y_test, y_pred, y_score=y_score, average=average)

    out = {
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'k': k,
        'metrics': metrics,
        'classifier': knn,
    }
    return out


def knn_crossval(embeddings, labels, k=5, n_splits=5, average='macro', n_jobs=1):
    """Perform Stratified K-fold cross-validated k-NN evaluation and return mean metrics."""
    X = _to_numpy(embeddings)
    y = _to_numpy(labels)
    if y.ndim == 2:
        y = np.argmax(y, axis=1)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_list = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=n_jobs)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        y_score = knn.predict_proba(X_test) if hasattr(knn, 'predict_proba') else None
        m = classification_metrics(y_test, y_pred, y_score=y_score, average=average)
        metrics_list.append(m)

    # aggregate
    agg = {}
    keys = metrics_list[0].keys()
    for k in keys:
        vals = [m[k] for m in metrics_list]
        # AUC may contain None
        if all(v is not None for v in vals):
            agg[k] = float(np.mean(vals))
        else:
            # for non-numeric (confusion matrix), skip
            try:
                agg[k] = float(np.mean([v for v in vals if v is not None]))
            except Exception:
                agg[k] = None

    agg['k'] = k
    agg['n_splits'] = n_splits
    return agg

