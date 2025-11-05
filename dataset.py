from pathlib import Path
from typing import Tuple, Optional, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config


def load_spectrum(filepath: str, modality: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load x and y data from file. Returns (x, y) tuple."""
    try:
        if modality == 'raman':
            # Load both x and y columns
            data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
            if data.ndim == 1:
                # Only y provided, create dummy x
                y = data
                x = np.arange(len(y), dtype=np.float32)
            else:
                x = data[:, 0]
                y = data[:, 1]
        else:  # gc
            # Skip comment lines and load x (column 1) and y (column 2)
            data = np.loadtxt(filepath, delimiter=',', dtype=np.float32, comments='#')
            if data.ndim == 1:
                y = data
                x = np.arange(len(y), dtype=np.float32)
            else:
                x = data[:, 1]  # Time column
                y = data[:, 2]  # Intensity column
        return x, y
    except:
        return None

def preprocess_spectrum(x: np.ndarray, y: np.ndarray, modality: str) -> np.ndarray:
    """
    Preprocess spectrum - both modalities interpolated to COMMON_LENGTH (4096):
    - Raman: Map to common x-range (300.1-3399.4), zero-pad missing regions, interpolate to 4096
    - GC: Interpolate from 5347 to 4096 (downsample slightly)
    """
    if modality == 'raman':
        # Create common x-axis for Raman (300.1 to 3399.4 cm^-1)
        common_x = np.linspace(config.RAMAN_X_MIN, config.RAMAN_X_MAX, config.COMMON_LENGTH)
        
        # Interpolate: areas without data will be 0 (left=0, right=0 for extrapolation)
        y_aligned = np.interp(common_x, x, y, left=0.0, right=0.0)
        
        # Normalize to [0, 1]
        y_min, y_max = y_aligned.min(), y_aligned.max()
        y_normalized = (y_aligned - y_min) / (y_max - y_min + 1e-10)
        
        return y_normalized.astype(np.float32)
    
    else:  # gc
        # GC: interpolate from original x-range to COMMON_LENGTH
        common_x = np.linspace(config.GC_X_MIN, config.GC_X_MAX, config.COMMON_LENGTH)
        y_aligned = np.interp(common_x, x, y)
        
        # Normalize to [0, 1]
        y_min, y_max = y_aligned.min(), y_aligned.max()
        y_normalized = (y_aligned - y_min) / (y_max - y_min + 1e-10)
        
        return y_normalized.astype(np.float32)


class SpectralDataset(Dataset):
    """Dataset for preloaded spectra."""
    def __init__(self, spectra: np.ndarray, class_labels: np.ndarray, modalities: np.ndarray):
        self.spectra = spectra.astype(np.float32)
        self.class_labels = class_labels.astype(np.float32)
        self.modalities = modalities.astype(np.float32)

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        spectrum = torch.from_numpy(self.spectra[idx])
        labels = torch.from_numpy(self.class_labels[idx])
        modality = torch.tensor([self.modalities[idx]], dtype=torch.float32)
        return spectrum, torch.cat([labels, modality])


class LazySpectralDataset(Dataset):
    """Lazy loading dataset - loads data on-the-fly."""
    def __init__(self, file_paths: List[str], class_labels: List[np.ndarray], modalities: List[str]):
        self.file_paths = file_paths
        self.class_labels = class_labels
        self.modalities_str = modalities  # Keep string version for reference
        # Convert to numeric for sampler
        self.modalities = np.array([0.0 if m == 'raman' else 1.0 for m in modalities], dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        result = load_spectrum(self.file_paths[idx], self.modalities_str[idx])
        if result is None:
            spectrum = torch.zeros(config.COMMON_LENGTH, dtype=torch.float32)
        else:
            x, y = result
            y_processed = preprocess_spectrum(x, y, self.modalities_str[idx])
            spectrum = torch.from_numpy(y_processed)
        
        labels = torch.from_numpy(self.class_labels[idx].astype(np.float32))
        modality = torch.tensor([self.modalities[idx]], dtype=torch.float32)
        return spectrum, torch.cat([labels, modality])

def load_file_metadata() -> Tuple[List[Dict], Dict]:
    """Scan directories and build file metadata."""
    all_files = []
    stats = {
        'raman_count': 0, 'gc_count': 0,
        'raman_classes': {cls: 0 for cls in config.CLASSES},
        'gc_classes': {cls: 0 for cls in config.CLASSES}
    }
    
    # Raman files
    for class_label, rel_dir in config.RAMAN_DIRS.items():
        full_dir = Path(config.BASE_DATA_DIR) / rel_dir
        if not full_dir.exists():
            continue
            
        for filepath in full_dir.glob('*.csv'):
            all_files.append({
                "path": str(filepath),
                "modality": 'raman',
                "classes": [class_label]
            })
            stats['raman_count'] += 1
            stats['raman_classes'][class_label] += 1
    
    # GC files
    for class_label, rel_dir in config.GC_DIRS.items():
        full_dir = Path(config.BASE_DATA_DIR) / rel_dir
        if not full_dir.exists():
            continue
            
        for filepath in full_dir.glob('*.csv'):
            all_files.append({
                "path": str(filepath),
                "modality": 'gc',
                "classes": [class_label]
            })
            stats['gc_count'] += 1
            stats['gc_classes'][class_label] += 1
    
    return all_files, stats

def process_single_file(file_info: Dict) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Process a single file."""
    class_to_idx = {cls: i for i, cls in enumerate(config.CLASSES)}
    
    try:
        result = load_spectrum(file_info["path"], file_info["modality"])
        if result is None:
            return None
        
        x, y = result
        y_normalized = preprocess_spectrum(x, y, file_info["modality"])
        
        class_vec = np.zeros(len(config.CLASSES), dtype=np.float32)
        for cls in file_info["classes"]:
            if cls in class_to_idx:
                class_vec[class_to_idx[cls]] = 1.0
        
        modality = 0.0 if file_info["modality"] == 'raman' else 1.0
        
        return y_normalized, class_vec, modality
        
    except:
        return None


def process_files(file_metadata: List[Dict], use_parallel: bool = True, 
                  max_workers: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process files and return spectra, class vectors, and modalities."""
    
    if use_parallel and len(file_metadata) > 50:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_file, file_info) 
                      for file_info in file_metadata]
            
            results = []
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result is not None:
                    results.append(result)
                
                if (i + 1) % 5000 == 0:
                    print(f"  Processed {i+1}/{len(futures)} files")
            
            if not results:
                raise ValueError("No files were successfully processed!")
            
            spectra, class_vectors, modalities = zip(*results)
    else:
        results = []
        for i, file_info in enumerate(file_metadata):
            result = process_single_file(file_info)
            if result is not None:
                results.append(result)
            
            if (i + 1) % 5000 == 0:
                print(f"  Processed {i+1}/{len(file_metadata)} files")
        
        if not results:
            raise ValueError("No files were successfully processed!")
        
        spectra, class_vectors, modalities = zip(*results)
    
    print(f"Successfully processed {len(results)}/{len(file_metadata)} files")
    return (np.array(spectra, dtype=np.float32), 
            np.array(class_vectors, dtype=np.float32), 
            np.array(modalities, dtype=np.float32))

def prepare_dataloader(use_lazy_loading: bool = False, 
                      use_parallel: bool = True,
                      max_workers: int = 4,
                      num_workers: int = 2) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Prepare train and validation dataloaders."""
    print("Scanning files...")
    file_metadata, stats = load_file_metadata()
    
    print(f"Found {stats['raman_count']} Raman, {stats['gc_count']} GC files")
    print(f"Class distribution:")
    print(f"  Raman: {stats['raman_classes']}")
    print(f"  GC: {stats['gc_classes']}")
    
    total_files = len(file_metadata)
    
    if use_lazy_loading:
        print(f"Using lazy loading (memory-efficient mode)")
        # Prepare metadata for lazy loading
        class_to_idx = {cls: i for i, cls in enumerate(config.CLASSES)}
        
        file_paths = []
        class_labels = []
        modalities = []
        
        for file_info in file_metadata:
            class_vec = np.zeros(len(config.CLASSES), dtype=np.float32)
            for cls in file_info["classes"]:
                if cls in class_to_idx:
                    class_vec[class_to_idx[cls]] = 1.0
            
            file_paths.append(file_info["path"])
            class_labels.append(class_vec)
            modalities.append(file_info["modality"])
        
        stratify_labels = [str(cv) for cv in class_labels]
        indices = np.arange(len(file_paths))
        train_idx, val_idx = train_test_split(
            indices, test_size=config.VAL_SPLIT, random_state=42, stratify=stratify_labels
        )
        
        train_dataset = LazySpectralDataset(
            [file_paths[i] for i in train_idx],
            [class_labels[i] for i in train_idx],
            [modalities[i] for i in train_idx]
        )
        val_dataset = LazySpectralDataset(
            [file_paths[i] for i in val_idx],
            [class_labels[i] for i in val_idx],
            [modalities[i] for i in val_idx]
        )
    else:
        print("Processing spectra...")
        spectra, class_vectors, modalities = process_files(
            file_metadata, use_parallel=use_parallel, max_workers=max_workers
        )
        
        if len(spectra) == 0:
            raise ValueError("No data loaded successfully")
        
        stratify_labels = [str(cv) for cv in class_vectors]
        indices = np.arange(len(spectra))
        train_idx, val_idx = train_test_split(
            indices, test_size=config.VAL_SPLIT, random_state=42, stratify=stratify_labels
        )
        
        train_dataset = SpectralDataset(
            spectra[train_idx], class_vectors[train_idx], modalities[train_idx]
        )
        val_dataset = SpectralDataset(
            spectra[val_idx], class_vectors[val_idx], modalities[val_idx]
        )
    
    # Use paired sampler for training (ensures cross-modal pairs in batch)
    from paired_sampler import CrossModalPairedSampler
    
    train_sampler = CrossModalPairedSampler(
        train_dataset,
        samples_per_class_per_modality=64,  # 64 samples per class per modality
        classes_per_batch=4,  # All 4 classes in each batch → 4 × 64 × 2 = 512 samples per batch
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=train_sampler,  # Use custom sampler instead of random
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    print(f"\n{'='*60}")
    print(f"Dataset ready: {len(train_dataset)} train, {len(val_dataset)} val samples")
    print(f"Memory mode: {'Lazy (on-the-fly)' if use_lazy_loading else 'Preloaded'}")
    print(f"DataLoader workers: {num_workers}")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader, config.CLASSES

