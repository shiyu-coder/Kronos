import os
import sys
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append("../")
from model import KronosTokenizer
from finetune_base_model import CustomKlineDataset
from config_loader import CustomFinetuneConfig


from finetune.utils.common import (
    setup_logging,
    create_dataloaders as _create_dataloaders,
    create_tokenizer_from_config,
)
from finetune.utils.training_utils import set_seed, get_model_size, format_time


def create_dataloaders(config):
    """Thin wrapper that passes CustomKlineDataset to the shared helper."""
    return _create_dataloaders(CustomKlineDataset, config)

def train_tokenizer(model, device, config, save_dir, logger):
    logger.info("Starting tokenizer training...")
    use_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0
    world_size = dist.get_world_size() if use_ddp else 1
    
    train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler = create_dataloaders(config)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.tokenizer_learning_rate,
        weight_decay=config.adam_weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.tokenizer_learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=config.tokenizer_epochs,
        pct_start=0.03,
        div_factor=10
    )
    
    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    best_val_loss = float("inf")
    batch_idx_global = 0
    
    accumulation_steps = getattr(config, 'accumulation_steps', 1)
    
    for epoch in range(config.tokenizer_epochs):
        epoch_start_time = time.time()
        model.train()
        
        train_dataset.set_epoch_seed(epoch * 10000)
        val_dataset.set_epoch_seed(0)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        for batch_idx, (ori_batch_x, _) in enumerate(train_loader):
            ori_batch_x = ori_batch_x.to(device, non_blocking=True)
            
            current_batch_total_loss = 0.0
            for j in range(accumulation_steps):
                start_idx = j * (ori_batch_x.shape[0] // accumulation_steps)
                end_idx = (j + 1) * (ori_batch_x.shape[0] // accumulation_steps)
                batch_x = ori_batch_x[start_idx:end_idx]
                
                zs, bsq_loss, _, _ = (model.module if use_ddp else model)(batch_x)
                z_pre, z = zs
                
                recon_loss_pre = F.mse_loss(z_pre, batch_x)
                recon_loss_all = F.mse_loss(z, batch_x)
                recon_loss = recon_loss_pre + recon_loss_all
                loss = (recon_loss + bsq_loss) / 2
                
                loss_scaled = loss / accumulation_steps
                current_batch_total_loss += loss.item()
                loss_scaled.backward()
            
            torch.nn.utils.clip_grad_norm_((model.module if use_ddp else model).parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            if (batch_idx_global + 1) % config.log_interval == 0:
                avg_loss = current_batch_total_loss / accumulation_steps
                lr = optimizer.param_groups[0]["lr"]
                log_msg = (f"[Epoch {epoch+1}/{config.tokenizer_epochs}, Step {batch_idx+1}/{len(train_loader)}] "
                          f"LR: {lr:.6f}, Loss: {avg_loss:.4f}")
                logger.info(log_msg)
                if rank == 0:
                    print(log_msg)
                
                detail_msg = (f"  - VQ Loss: {bsq_loss.item():.4f}\n"
                            f"  - Recon Loss Pre: {recon_loss_pre.item():.4f}\n"
                            f"  - Recon Loss All: {recon_loss_all.item():.4f}")
                logger.info(detail_msg)
                if rank == 0:
                    print(detail_msg)
            
            batch_idx_global += 1
        
        model.eval()
        tot_val_loss_sum_rank = 0.0
        val_sample_count_rank = 0
        
        with torch.no_grad():
            for ori_batch_x, _ in val_loader:
                ori_batch_x = ori_batch_x.to(device, non_blocking=True)
                zs, _, _, _ = (model.module if use_ddp else model)(ori_batch_x)
                _, z = zs
                val_loss_item = F.mse_loss(z, ori_batch_x)
                
                tot_val_loss_sum_rank += val_loss_item.item() * ori_batch_x.size(0)
                val_sample_count_rank += ori_batch_x.size(0)
        
        if use_ddp:
            tensor_sum = torch.tensor([tot_val_loss_sum_rank, val_sample_count_rank], dtype=torch.float64, device=device)
            dist.all_reduce(tensor_sum, op=dist.ReduceOp.SUM)
            tot_val_loss_all = tensor_sum[0].item()
            val_count_all = int(tensor_sum[1].item())
            avg_val_loss = (tot_val_loss_all / val_count_all) if val_count_all > 0 else 0.0
        else:
            avg_val_loss = tot_val_loss_sum_rank / val_sample_count_rank if val_sample_count_rank > 0 else 0
        
        epoch_time = time.time() - epoch_start_time
        epoch_summary = (f"\n--- Epoch {epoch+1}/{config.tokenizer_epochs} Summary ---\n"
                       f"Validation Loss: {avg_val_loss:.4f}\n"
                       f"Epoch Time: {format_time(epoch_time)}\n"
                       f"Total Training Time: {format_time(time.time() - epoch_start_time)}\n")
        logger.info(epoch_summary)
        if rank == 0:
            print(epoch_summary)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if rank == 0:
                model_save_path = os.path.join(save_dir, "best_model")
                os.makedirs(model_save_path, exist_ok=True)
                (model.module if use_ddp else model).save_pretrained(model_save_path)
                save_msg = f"Best model saved to: {model_save_path} (validation loss: {best_val_loss:.4f})"
                logger.info(save_msg)
                print(save_msg)
    
    return best_val_loss


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Kronos Tokenizer Fine-tuning Training')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Configuration file path (default: config.yaml)')
    args = parser.parse_args()
    
    config = CustomFinetuneConfig(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(config.tokenizer_save_path, exist_ok=True)
    
    log_dir = os.path.join(config.base_save_path, "logs")
    logger = setup_logging("tokenizer_training", log_dir, 0)
    
    set_seed(config.seed)
    
    # 加载预训练tokenizer
    # Load pretrained tokenizer or random init
    if getattr(config, 'pre_trained_tokenizer', True):
        logger.info("Loading pretrained tokenizer...")
        print("Loading pretrained tokenizer...")
        tokenizer = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
    else:
        print("pre_trained_tokenizer=False, randomly initializing Tokenizer architecture")
        tokenizer = create_tokenizer_from_config(config.pretrained_tokenizer_path)
    tokenizer = tokenizer.to(device)
    
    model_size = get_model_size(tokenizer)
    logger.info(f"Tokenizer parameters: {model_size}")
    print(f"Tokenizer parameters: {model_size}")
    
    logger.info("=== Training Configuration ===")
    logger.info(f"Data path: {config.data_path}")
    logger.info(f"Lookback window: {config.lookback_window}")
    logger.info(f"Predict window: {config.predict_window}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.tokenizer_learning_rate}")
    logger.info(f"Training epochs: {config.tokenizer_epochs}")
    logger.info(f"Device: {device}")
    logger.info(f"Distributed training: False")
    
    logger.info("Starting tokenizer fine-tuning training...")
    print("Starting tokenizer fine-tuning training...")
    best_val_loss = train_tokenizer(tokenizer, device, config, config.tokenizer_save_path, logger)
    
    final_msg = f"Tokenizer training completed! Best validation loss: {best_val_loss:.4f}\nModel saved to: {config.tokenizer_save_path}"
    logger.info(final_msg)
    print(final_msg)


if __name__ == "__main__":
    main()
    
