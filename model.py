"""
Cross-Modal Contrastive Learning Model for Raman and GC Spectroscopy
(MODIFIED: Added Peak Apex Feature Concatenation for GC Encoder)
(REFACTORED: Removed all temperature/loss logic from the model)

이 모델은:
1. Raman과 GC 데이터를 공통 임베딩 공간으로 매핑
2. [NEW] GC Encoder는 "원본 신호"와 "피크 꼭대기 신호" 2-채널 입력을 사용
3. [REMOVED] Temperature logic is now in the loss function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config


class PeakApexFeatureExtractor(nn.Module):
    """
    [NEW CLASS]
    비학습형(non-trainable) 레이어.
    2차 도함수(Laplacian)를 이용해 피크의 꼭대기(Apex)를 감지하는
    특징 벡터(Channel)를 생성합니다.
    """
    def __init__(self):
        super().__init__()
        # 2차 도함수 근사 필터: [1, -2, 1]
        laplacian_kernel = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float32).view(1, 1, 3)
        
        # Conv1d 레이어로 구현
        self.conv_laplacian = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        
        # 필터 가중치 고정 (학습 X)
        self.conv_laplacian.weight.data = laplacian_kernel
        self.conv_laplacian.weight.requires_grad = False
        
    def forward(self, x):
        """
        Args:
            x: (B, 1, L) 원본 스펙트럼 신호
        Returns:
            peak_signal: (B, 1, L) 피크 꼭대기 감지 신호 (0~1 정규화)
        """
        # 2차 도함수 계산 (Concavity)
        # 피크 꼭대기(볼록)는 음수 값을 가짐
        concavity = self.conv_laplacian(x)
        
        # 음수 concavity를 양수로 뒤집고(피크 강조), ReLU로 양수 부분만 남김
        peak_signal = F.relu(-concavity)
        
        # [Batch, Channel, Time]에서 Time 축 기준 max로 정규화
        # (B, 1, L) -> (B, 1, 1)
        batch_max = torch.max(peak_signal, dim=2, keepdim=True)[0]
        
        # 0으로 나누는 것을 방지
        eps = 1e-6 
        
        # 0~1 사이 값으로 정규화
        peak_signal_norm = peak_signal / (batch_max + eps)
        
        return peak_signal_norm


class ResidualBlock1D(nn.Module):
    """1D Residual Block for better gradient flow"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class SpectralEncoder1D(nn.Module):
    """
    Enhanced 1D CNN Encoder for Spectral Data
    (MODIFIED: 이제 (B, C, L) 입력을 바로 받음)
    """
    def __init__(self, in_channels=1, hidden_dim=config.HIDDEN_DIM, embed_dim=config.EMBEDDING_DIM):
        super().__init__()
        
        # Stage 1: Initial feature extraction
        self.stage1 = nn.Sequential(
            # `in_channels`가 1 (Raman) 또는 2 (GC)가 될 수 있음
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Stage 2: Feature enhancement
        self.stage2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            ResidualBlock1D(128, kernel_size=3),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Stage 3: Deep features
        self.stage3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            ResidualBlock1D(256, kernel_size=3),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Stage 4: High-level features
        self.stage4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            ResidualBlock1D(512, kernel_size=3),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Stage 5: Abstract features
        self.stage5 = nn.Sequential(
            nn.Conv1d(512, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            ResidualBlock1D(hidden_dim, kernel_size=3),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Projection head - 3-layer MLP
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 2, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L) where C=1 for Raman, C=2 for GC
        Returns:
            embedding: (B, embed_dim) L2-normalized
        """
        # [REMOVED] x = x.unsqueeze(1) <-- 이제 입력이 (B, C, L)
        
        # Feature extraction
        x = self.stage1(x)  # (B, 64, 2048)
        x = self.stage2(x)  # (B, 128, 1024)
        x = self.stage3(x)  # (B, 256, 512)
        x = self.stage4(x)  # (B, 512, 256)
        x = self.stage5(x)  # (B, hidden_dim, 128)
        
        # Global pooling
        x = self.global_pool(x)  # (B, hidden_dim, 1)
        x = x.squeeze(-1)  # (B, hidden_dim)
        
        # Project to embedding space
        embedding = self.projection_head(x)  # (B, embed_dim)
        
        # L2 normalize - critical for contrastive learning
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding


class CrossModalContrastiveModel(nn.Module):
    """
    Cross-Modal Contrastive Learning Model (CLIP-style for Spectroscopy)
    
    (REFACTORED)
    Architecture:
    - Dual encoders:
      - Raman (1-channel input)
      - GC (2-channel input: original + peak_apex)
    - Shared embedding space
    - [REMOVED] Temperature logic.
    """
    
    def __init__(self, hidden_dim=config.HIDDEN_DIM, embed_dim=config.EMBEDDING_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        
        # --- [MODIFIED] ---
        # 1. Raman Encoder (1-channel input)
        self.raman_encoder = SpectralEncoder1D(
            in_channels=1, 
            hidden_dim=hidden_dim, 
            embed_dim=embed_dim
        )
        
        # 2. GC Encoder (2-channel input)
        self.gc_encoder = SpectralEncoder1D(
            in_channels=2, # <-- 채널 2개 (Original + PeakApex)
            hidden_dim=hidden_dim, 
            embed_dim=embed_dim
        )
        
        # 3. [NEW] Peak Apex Feature Extractor
        self.peak_feature_extractor = PeakApexFeatureExtractor()
        
    
    def encode_raman(self, spectra: torch.Tensor) -> torch.Tensor:
        """Encode Raman spectra to embeddings"""
        # (B, 1, L) -> (B, embed_dim)
        return self.raman_encoder(spectra)
    
    def encode_gc(self, spectra: torch.Tensor) -> torch.Tensor:
        """Encode GC spectra to embeddings"""
        # (B, 2, L) -> (B, embed_dim)
        return self.gc_encoder(spectra)
    
    def forward(self, spectra: torch.Tensor, modalities: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - routes data to appropriate encoder
        
        (MODIFIED: GC path now creates 2-channel input)
        
        Args:
            spectra: (B, 4096) spectral data
            modalities: (B,) modality indicator (0.0=Raman, 1.0=GC)
        
        Returns:
            embeddings: (B, embed_dim) L2-normalized embeddings
        """
        if modalities.dim() > 1:
            modalities = modalities.squeeze(-1)
        
        raman_mask = (modalities == 0.0)
        gc_mask = (modalities == 1.0)
        
        embeddings = torch.zeros(
            spectra.size(0), self.embed_dim, 
            device=spectra.device, dtype=spectra.dtype
        )
        
        # --- [MODIFIED] Raman Path ---
        if raman_mask.any():
            # (B_raman, 4096) -> (B_raman, 1, 4096)
            raman_input = spectra[raman_mask].unsqueeze(1)
            embeddings[raman_mask] = self.encode_raman(raman_input)
        
        # --- [MODIFIED] GC Path ---
        if gc_mask.any():
            # (B_gc, 4096) -> (B_gc, 1, 4096)
            gc_input_orig = spectra[gc_mask].unsqueeze(1)
            
            # [NEW] Create peak feature channel
            # (B_gc, 1, 4096) -> (B_gc, 1, 4096)
            gc_input_peak = self.peak_feature_extractor(gc_input_orig)
            
            # [NEW] Concatenate channels
            # (B_gc, 1, 4096) + (B_gc, 1, 4096) -> (B_gc, 2, 4096)
            gc_input_combined = torch.cat([gc_input_orig, gc_input_peak], dim=1)
            
            embeddings[gc_mask] = self.encode_gc(gc_input_combined)
        
        return embeddings
    
    # --- [REFACTORED] ---
    # get_temperature() and compute_similarity_matrix() have been
    # removed. This logic now lives in the loss function.
    # --- [END REFACTORED] ---