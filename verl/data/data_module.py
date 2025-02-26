import os
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from verl.common.distributed_utils import get_rank, get_world_size


class TextDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        max_prompt_length: int,
        max_response_length: int,
    ) -> None:
        self.df = pd.read_parquet(file_path)
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        prompt = row["prompt"]
        response = row["response"]
        return {
            "prompt": prompt,
            "response": response,
        }


class DataModule:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self) -> None:
        # Create train dataset
        if isinstance(self.cfg.train_files, str):
            train_files = [self.cfg.train_files]
        else:
            train_files = self.cfg.train_files
        
        self.train_dataset = TextDataset(
            file_path=train_files[0],
            max_prompt_length=self.cfg.max_prompt_length,
            max_response_length=self.cfg.max_response_length,
        )

        # Create validation dataset
        if isinstance(self.cfg.val_files, str):
            val_files = [self.cfg.val_files]
        else:
            val_files = self.cfg.val_files
        
        self.val_dataset = TextDataset(
            file_path=val_files[0],
            max_prompt_length=self.cfg.max_prompt_length,
            max_response_length=self.cfg.max_response_length,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train_batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers if hasattr(self.cfg, "num_workers") else 0,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers if hasattr(self.cfg, "num_workers") else 0,
            pin_memory=True,
        ) 