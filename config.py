"""Configuration for transfer learning script.

Edit hyperparameters here — no argparse is used per project requirements.
"""
# Paths
CHECKPOINT_PATH = 'models/ContrastiveModel_ResNet.pth'
SAVE_DIR = 'models/transfer_contrastive_classifier.pth'

DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'




BASE_DATA_DIR = './data'

CLASSES = [
    "2-CEES", "2-CEPS", "DMMP", "4-NP"
]

CROSS_MODAL_CLASSES = ['2-CEES', '2-CEPS', 'DMMP', '4-NP']

RAMAN_DIRS = {
    "2-CEES": "Raman_single/CEES",
    "2-CEPS": "Raman_single/CEPS",
    "DMMP": "Raman_single/DMMP",
    "4-NP": "Raman_single/4NP"
}

GC_DIRS = {
    "2-CEES": "GC_single/CEES",
    "2-CEPS": "GC_single/CEPS",
    "DMMP": "GC_single/DMMP",
    "4-NP": "GC_single/4NP"
}


VAL_SPLIT = 0.2

# Training parameters
BATCH_SIZE = 512  # Effective batch size from sampler (4 classes × 64 samples × 2 modalities)
EPOCHS = 2000
LR = 1e-5
WEIGHT_DECAY = 1e-4  # Increased from 1e-5 for stronger regularization

# Model Architecture
EMBEDDING_DIM = 512  # Larger embedding dimension
HIDDEN_DIM = 1024    # Larger hidden dimension for better representation


# Data preprocessing parameters
COMMON_LENGTH = 4096  # Both Raman and GC interpolated to this length

# Raman X-axis range - GLOBAL range covering ALL files
# Files will be mapped to this grid, zero-padded where no data exists
RAMAN_X_MIN = 300.0      # cm^-1 (minimum across all Raman files)
RAMAN_X_MAX = 3400.0     # cm^-1 (maximum across all Raman files)

# GC has consistent range across all classes
GC_X_MIN = 0.0910        # minutes (retention time)
GC_X_MAX = 16.9960       # minutes (retention time)


# Whether to use lazy loading (passes through existing dataset.prepare_dataloader)
USE_LAZY_LOADING = True


# Data Loading Configuration
USE_LAZY_LOADING = True  # MUST be True for 200k files
USE_PARALLEL = True  # Use parallel preprocessing for faster loading
MAX_WORKERS = 16  # For preprocessing
NUM_WORKERS = 16  # DataLoader workers - reduced to save memory 
