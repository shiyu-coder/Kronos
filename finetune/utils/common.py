"""
Shared utilities for Kronos finetune scripts.

Consolidates duplicated helpers from finetune_csv/ into a single module:
- setup_logging: configurable logger with rotating file handler
- create_dataloaders: build train/val DataLoaders (with optional DDP)
- create_tokenizer_from_config: random-init a KronosTokenizer from config.json
- create_predictor_from_config: random-init a Kronos predictor from config.json
"""

import os
import json
import datetime
import logging
from logging.handlers import RotatingFileHandler

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def setup_logging(name, log_dir, rank=0):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"{name}_rank_{rank}")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    log_file = os.path.join(log_dir, f"{name}_rank_{rank}.log")
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    console_handler = None
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    if console_handler is not None:
        console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if console_handler is not None:
        logger.addHandler(console_handler)
    logger.info(f"=== {name} Started ===")
    logger.info(f"Log Directory: {log_dir}")
    logger.info(f"Rank: {rank}")
    logger.info(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return logger


def create_dataloaders(dataset_class, config):
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print("Creating data loaders...")
    train_dataset = dataset_class(
        data_path=config.data_path, data_type="train",
        lookback_window=config.lookback_window, predict_window=config.predict_window,
        clip=config.clip, seed=config.seed,
        train_ratio=config.train_ratio, val_ratio=config.val_ratio, test_ratio=config.test_ratio,
    )
    val_dataset = dataset_class(
        data_path=config.data_path, data_type="val",
        lookback_window=config.lookback_window, predict_window=config.predict_window,
        clip=config.clip, seed=config.seed + 1,
        train_ratio=config.train_ratio, val_ratio=config.val_ratio, test_ratio=config.test_ratio,
    )
    use_ddp = dist.is_available() and dist.is_initialized()
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False, drop_last=False) if use_ddp else None
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=(train_sampler is None), num_workers=config.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True, drop_last=False, sampler=val_sampler)
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    return train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler


def create_tokenizer_from_config(config_path):
    from model import KronosTokenizer
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, "config.json")
    with open(config_path, "r") as f:
        arch = json.load(f)
    return KronosTokenizer(
        d_in=arch.get("d_in", 6), d_model=arch.get("d_model", 256),
        n_heads=arch.get("n_heads", 4), ff_dim=arch.get("ff_dim", 512),
        n_enc_layers=arch.get("n_enc_layers", 4), n_dec_layers=arch.get("n_dec_layers", 4),
        ffn_dropout_p=arch.get("ffn_dropout_p", 0.0), attn_dropout_p=arch.get("attn_dropout_p", 0.0),
        resid_dropout_p=arch.get("resid_dropout_p", 0.0),
        s1_bits=arch.get("s1_bits", 10), s2_bits=arch.get("s2_bits", 10),
        beta=arch.get("beta", 0.05), gamma0=arch.get("gamma0", 1.0),
        gamma=arch.get("gamma", 1.1), zeta=arch.get("zeta", 0.05),
        group_size=arch.get("group_size", 4),
    )


def create_predictor_from_config(config_path):
    from model import Kronos
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, "config.json")
    with open(config_path, "r") as f:
        arch = json.load(f)
    return Kronos(
        s1_bits=arch.get("s1_bits", 10), s2_bits=arch.get("s2_bits", 10),
        n_layers=arch.get("n_layers", 12), d_model=arch.get("d_model", 832),
        n_heads=arch.get("n_heads", 16), ff_dim=arch.get("ff_dim", 2048),
        ffn_dropout_p=arch.get("ffn_dropout_p", 0.2), attn_dropout_p=arch.get("attn_dropout_p", 0.0),
        resid_dropout_p=arch.get("resid_dropout_p", 0.2), token_dropout_p=arch.get("token_dropout_p", 0.0),
        learn_te=arch.get("learn_te", True),
    )
