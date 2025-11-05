"""
Cross-Modal Paired Sampler for Contrastive Learning

이 샘플러는 각 배치에서:
1. 같은 클래스의 Raman과 GC 데이터를 균등하게 샘플링
2. Positive/Negative 비율을 일정하게 유지
3. 각 배치가 contrastive learning에 최적화되도록 구성
"""

import torch
import numpy as np
from torch.utils.data import Sampler
from typing import List, Iterator
from collections import defaultdict


class CrossModalPairedSampler(Sampler):
    """
    Cross-modal contrastive learning을 위한 balanced batch sampler.
    
    Strategy:
    - 각 배치에 K개의 클래스 포함
    - 각 클래스당 N개의 Raman + N개의 GC 샘플 포함
    - 결과: 배치 크기 = K * N * 2 (클래스 수 * 샘플 수 * 모달리티 수)
    
    Example:
        K=2 클래스, N=16 샘플/클래스/모달리티
        → 배치 크기 = 2 * 16 * 2 = 64
        → 각 배치에 32개 positive pairs (같은 클래스 내 cross-modal)
    """
    
    def __init__(self, 
                 dataset,
                 samples_per_class_per_modality: int = 16,
                 classes_per_batch: int = 2,
                 shuffle: bool = True,
                 seed: int = 42):
        """
        Args:
            dataset: Dataset with class_labels and modalities attributes
            samples_per_class_per_modality: 각 클래스당 각 모달리티에서 뽑을 샘플 수
            classes_per_batch: 각 배치에 포함할 클래스 수
            shuffle: 에폭마다 섞을지 여부
            seed: Random seed
        """
        self.dataset = dataset
        self.samples_per_class = samples_per_class_per_modality
        self.classes_per_batch = classes_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # 데이터셋의 클래스와 모달리티별로 인덱스 그룹화
        self._build_class_modality_indices()
        
        # 배치 크기 계산
        self.batch_size = classes_per_batch * samples_per_class_per_modality * 2
        
        # 총 배치 수 계산 (데이터를 최대한 활용)
        self.num_batches = self._calculate_num_batches()
        
        print(f"Sampler initialized:")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Classes per batch: {classes_per_batch}")
        print(f"  - Samples per class per modality: {samples_per_class_per_modality}")
        print(f"  - Estimated batches per epoch: {self.num_batches}")
        print(f"  - Class distribution: {[(cls, len(indices['raman']), len(indices['gc'])) for cls, indices in self.class_indices.items()]}")
    
    def _build_class_modality_indices(self):
        """데이터를 (클래스, 모달리티)별로 그룹화"""
        self.class_indices = defaultdict(lambda: {'raman': [], 'gc': []})
        
        for idx in range(len(self.dataset)):
            # LazyDataset과 SpectralDataset 모두 지원
            if hasattr(self.dataset, 'class_labels') and hasattr(self.dataset, 'modalities'):
                class_label = self.dataset.class_labels[idx]
                modality = self.dataset.modalities[idx]
                
                # One-hot → class index
                if isinstance(class_label, np.ndarray):
                    class_idx = np.argmax(class_label)
                else:
                    class_idx = int(class_label)
                
                # Modality: convert to float to handle different types
                modality_value = float(modality)
                modality_str = 'raman' if modality_value < 0.5 else 'gc'
            else:
                # Subset인 경우 - __getitem__ 호출
                _, labels_and_mod = self.dataset[idx]
                class_label = labels_and_mod[:-1].numpy()
                modality = labels_and_mod[-1].item()
                
                class_idx = np.argmax(class_label)
                modality_str = 'raman' if float(modality) < 0.5 else 'gc'
            
            self.class_indices[class_idx][modality_str].append(idx)
        
        # 각 클래스별로 인덱스 배열로 변환
        for class_idx in self.class_indices:
            self.class_indices[class_idx]['raman'] = np.array(self.class_indices[class_idx]['raman'])
            self.class_indices[class_idx]['gc'] = np.array(self.class_indices[class_idx]['gc'])
        
        self.num_classes = len(self.class_indices)
    
    def _calculate_num_batches(self):
        """에폭당 배치 수 계산"""
        # 각 클래스에서 뽑을 수 있는 최대 배치 수 계산
        min_samples_per_class = float('inf')
        
        for class_idx, indices in self.class_indices.items():
            raman_count = len(indices['raman'])
            gc_count = len(indices['gc'])
            
            # 각 모달리티에서 최소한 samples_per_class개씩 필요
            max_batches_this_class = min(
                raman_count // self.samples_per_class,
                gc_count // self.samples_per_class
            )
            min_samples_per_class = min(min_samples_per_class, max_batches_this_class)
        
        # 전체 배치 수: (클래스 조합 수) * (클래스당 최소 배치 수)
        # 간단히 하기 위해 모든 클래스를 순환하며 사용
        num_batches = (self.num_classes // self.classes_per_batch) * min_samples_per_class
        
        return max(1, num_batches)
    
    def __iter__(self) -> Iterator[List[int]]:
        """배치 인덱스 생성"""
        if self.shuffle:
            # 에폭마다 다른 시드 사용
            rng = np.random.RandomState(self.seed + self.epoch)
        else:
            rng = np.random.RandomState(self.seed)
        
        # 모든 클래스 ID
        all_classes = list(self.class_indices.keys())
        
        # 배치 생성
        for _ in range(self.num_batches):
            batch_indices = []
            
            # 이번 배치에 사용할 클래스 선택
            selected_classes = rng.choice(all_classes, size=self.classes_per_batch, replace=False)
            
            for class_idx in selected_classes:
                # Raman 샘플 선택
                raman_indices = self.class_indices[class_idx]['raman']
                if len(raman_indices) >= self.samples_per_class:
                    selected_raman = rng.choice(raman_indices, size=self.samples_per_class, replace=False)
                else:
                    # 부족하면 replacement
                    selected_raman = rng.choice(raman_indices, size=self.samples_per_class, replace=True)
                
                # GC 샘플 선택
                gc_indices = self.class_indices[class_idx]['gc']
                if len(gc_indices) >= self.samples_per_class:
                    selected_gc = rng.choice(gc_indices, size=self.samples_per_class, replace=False)
                else:
                    selected_gc = rng.choice(gc_indices, size=self.samples_per_class, replace=True)
                
                batch_indices.extend(selected_raman)
                batch_indices.extend(selected_gc)
            
            # 배치 내에서 섞기 (선택적)
            if self.shuffle:
                rng.shuffle(batch_indices)
            
            yield batch_indices
        
        self.epoch += 1
    
    def __len__(self) -> int:
        """총 배치 수"""
        return self.num_batches
    
    def set_epoch(self, epoch: int):
        """에폭 설정 (DistributedSampler 스타일)"""
        self.epoch = epoch
