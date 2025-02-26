import os
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from omegaconf import DictConfig

from verl.common.distributed_utils import get_rank, get_world_size, is_main_process
from verl.common.logging_utils import Logger
from verl.common.utils import get_timestamp
from verl.data.data_module import DataModule
from verl.rl.ppo.ppo_algorithm import PPOAlgorithm


class PPOTrainer:
    def __init__(self, cfg: DictConfig, loggers: List[Logger]) -> None:
        self.cfg = cfg
        self.loggers = loggers
        self.algorithm = PPOAlgorithm(cfg)
        self.epoch = 0
        self.step = 0
        self.val_step = 0
        self.best_val_reward = float("-inf")
        self.timestamp = get_timestamp()

    def fit(self, data_module: DataModule) -> None:
        # Prepare data
        data_module.prepare_data()
        train_dataloader = data_module.train_dataloader()
        val_dataloader = data_module.val_dataloader()

        # Validate before train
        if self.cfg.trainer.val_before_train:
            self.validate(val_dataloader)

        # Train
        for epoch in range(self.cfg.trainer.total_epochs):
            self.epoch = epoch
            self.train_epoch(train_dataloader)

            # Save checkpoint
            if (
                is_main_process()
                and self.epoch % self.cfg.trainer.save_freq == 0
                and self.epoch > 0
            ):
                self.save_checkpoint()

            # Validate
            if self.epoch % self.cfg.trainer.test_freq == 0:
                self.validate(val_dataloader)

        # Final validation
        self.validate(val_dataloader)

    def train_epoch(self, train_dataloader) -> None:
        # Set to train mode
        self.algorithm.train()

        # Train
        for batch in train_dataloader:
            # Train step
            metrics = self.algorithm.train_step(batch)

            # Log
            if is_main_process() and self.step % self.cfg.trainer.log_freq == 0:
                self.log(metrics, "train")

            # Update step
            self.step += 1

    def validate(self, val_dataloader) -> None:
        # Set to eval mode
        self.algorithm.eval()

        # Validate
        metrics_list = []
        for batch in val_dataloader:
            # Validation step
            metrics = self.algorithm.validation_step(batch)
            metrics_list.append(metrics)

        # Aggregate metrics
        metrics = self.algorithm.validation_epoch_end(metrics_list)

        # Log
        if is_main_process():
            self.log(metrics, "val")

        # Update best validation reward
        if is_main_process() and metrics["reward"] > self.best_val_reward:
            self.best_val_reward = metrics["reward"]
            self.save_checkpoint(is_best=True)

        # Update validation step
        self.val_step += 1

    def log(self, metrics: Dict[str, float], prefix: str) -> None:
        # Add prefix to metrics
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Add epoch and step to metrics
        metrics["epoch"] = self.epoch
        metrics["step"] = self.step if prefix == "train" else self.val_step

        # Log metrics
        for logger in self.loggers:
            logger.log_metrics(metrics)

    def save_checkpoint(self, is_best: bool = False) -> None:
        # Create checkpoint directory
        ckpt_dir = os.path.join(
            self.cfg.trainer.output_dir, self.cfg.trainer.experiment_name, "checkpoints"
        )
        os.makedirs(ckpt_dir, exist_ok=True)

        # Create checkpoint path
        ckpt_path = os.path.join(
            ckpt_dir, f"epoch_{self.epoch}_step_{self.step}_{self.timestamp}.pt"
        )
        if is_best:
            ckpt_path = os.path.join(ckpt_dir, f"best_{self.timestamp}.pt")

        # Save checkpoint
        checkpoint = {
            "epoch": self.epoch,
            "step": self.step,
            "val_step": self.val_step,
            "best_val_reward": self.best_val_reward,
            "algorithm": self.algorithm.state_dict(),
        }
        torch.save(checkpoint, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    def load_checkpoint(self, ckpt_path: str) -> None:
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.val_step = checkpoint["val_step"]
        self.best_val_reward = checkpoint["best_val_reward"]
        self.algorithm.load_state_dict(checkpoint["algorithm"])
        print(f"Loaded checkpoint from {ckpt_path}") 