import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist

from model import Kronos, KronosTokenizer, KronosPredictor

from config_loader import CustomFinetuneConfig
from finetune_tokenizer import train_tokenizer, set_seed, setup_logging as setup_tokenizer_logging
from finetune_base_model import train_model, create_dataloaders, setup_logging as setup_basemodel_logging


class SequentialTrainer:
    
    def __init__(self, config_path: str = None):
        self.config = CustomFinetuneConfig(config_path)
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(self.config.device_id if hasattr(self.config, 'device_id') else 0)))
        self.device = self._setup_device()
        
        self.config.print_config_summary()
    
    def _setup_device(self):
        if self.config.use_cuda and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            device = torch.device(f"cuda:{self.local_rank}")
        else:
            device = torch.device("cpu")
        
        if self.rank == 0:
            print(f"Using device: {device} (rank={self.rank}, world_size={self.world_size}, local_rank={self.local_rank})")
        return device
    
    def _setup_distributed(self):
        if self.world_size > 1 and torch.cuda.is_available():
            backend = os.environ.get("DIST_BACKEND", "nccl").lower()
            if not dist.is_initialized():
                dist.init_process_group(backend=backend)
            if self.rank == 0:
                print(f"Distributed training initialized: backend={backend}, world_size={self.world_size}")
        else:
            if self.rank == 0:
                print("Distributed training not enabled, using single GPU/CPU training")
    
    def _check_existing_models(self):
        tokenizer_exists = os.path.exists(self.config.tokenizer_best_model_path)
        basemodel_exists = os.path.exists(self.config.basemodel_best_model_path)
        
        print(f"Tokenizer model exists: {tokenizer_exists}")
        print(f"Basemodel model exists: {basemodel_exists}")
        
        return tokenizer_exists, basemodel_exists
    
    def _create_directories(self):
        os.makedirs(self.config.tokenizer_save_path, exist_ok=True)
        os.makedirs(self.config.basemodel_save_path, exist_ok=True)
        print(f"Created directory: {self.config.tokenizer_save_path}")
        print(f"Created directory: {self.config.basemodel_save_path}")
    
    def train_tokenizer_phase(self):
        print("\n" + "="*60)
        print("Starting Tokenizer Fine-tuning Phase")
        print("="*60)
        
        tokenizer_exists, _ = self._check_existing_models()
        if tokenizer_exists and self.config.skip_existing:
            print("Tokenizer model already exists, skipping training")
            return True
        
        log_dir = os.path.join(self.config.base_save_path, "logs")
        logger = setup_tokenizer_logging(self.config.exp_name, log_dir, self.rank)
        
        set_seed(self.config.seed)
        
        if getattr(self.config, 'pre_trained_tokenizer', True):
            logger.info("Loading pretrained tokenizer...")
            if self.rank == 0:
                print("Loading pretrained tokenizer...")
            tokenizer = KronosTokenizer.from_pretrained(self.config.pretrained_tokenizer_path)
        else:
            if self.rank == 0:
                print("pre_trained_tokenizer=False, randomly initializing Tokenizer architecture")
            import json
            cfg_path = os.path.join(self.config.pretrained_tokenizer_path, 'config.json')
            with open(cfg_path, 'r') as f:
                arch = json.load(f)
            tokenizer = KronosTokenizer(
                d_in=arch.get('d_in', 6),
                d_model=arch.get('d_model', 256),
                n_heads=arch.get('n_heads', 4),
                ff_dim=arch.get('ff_dim', 512),
                n_enc_layers=arch.get('n_enc_layers', 4),
                n_dec_layers=arch.get('n_dec_layers', 4),
                ffn_dropout_p=arch.get('ffn_dropout_p', 0.0),
                attn_dropout_p=arch.get('attn_dropout_p', 0.0),
                resid_dropout_p=arch.get('resid_dropout_p', 0.0),
                s1_bits=arch.get('s1_bits', 10),
                s2_bits=arch.get('s2_bits', 10),
                beta=arch.get('beta', 0.05),
                gamma0=arch.get('gamma0', 1.0),
                gamma=arch.get('gamma', 1.1),
                zeta=arch.get('zeta', 0.05),
                group_size=arch.get('group_size', 4)
            )
        tokenizer = tokenizer.to(self.device)
        
        model_size = sum(p.numel() for p in tokenizer.parameters())
        logger.info(f"Tokenizer parameters: {model_size:,}")
        if self.rank == 0:
            print(f"Tokenizer parameters: {model_size:,}")
        
        logger.info("=== Training Configuration ===")
        logger.info(f"Data path: {self.config.data_path}")
        logger.info(f"Lookback window: {self.config.lookback_window}")
        logger.info(f"Predict window: {self.config.predict_window}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.tokenizer_learning_rate}")
        logger.info(f"Training epochs: {self.config.tokenizer_epochs}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Distributed training: False")
        
        logger.info("Starting tokenizer fine-tuning training...")
        if self.rank == 0:
            print("Starting tokenizer fine-tuning training...")
        start_time = time.time()
        best_val_loss = train_tokenizer(
            tokenizer,
            self.device,
            self.config,
            self.config.tokenizer_save_path,
            logger,
        )
        training_time = time.time() - start_time
        
        final_msg = f"Tokenizer training completed! Best validation loss: {best_val_loss:.4f}\nTraining time: {training_time/60:.2f} minutes\nModel saved to: {self.config.tokenizer_save_path}"
        logger.info(final_msg)
        if self.rank == 0:
            print(f"\n{final_msg}")
        
        return True
    
    def train_basemodel_phase(self):
        print("\n" + "="*60)
        print("Starting Basemodel Fine-tuning Phase")
        print("="*60)
        
        if getattr(self.config, 'pre_trained_tokenizer', True):
            if not os.path.exists(self.config.finetuned_tokenizer_path):
                raise FileNotFoundError(f"Fine-tuned tokenizer does not exist: {self.config.finetuned_tokenizer_path}")
        
        _, basemodel_exists = self._check_existing_models()
        if basemodel_exists and self.config.skip_existing:
            print("Basemodel model already exists, skipping training")
            return True
        
        log_dir = os.path.join(self.config.base_save_path, "logs")
        logger = setup_basemodel_logging(self.config.exp_name, log_dir, self.rank)
        
        set_seed(self.config.seed)
        
        if getattr(self.config, 'pre_trained_tokenizer', True):
            logger.info("Loading fine-tuned tokenizer...")
            if self.rank == 0:
                print("Loading fine-tuned tokenizer...")
            tokenizer = KronosTokenizer.from_pretrained(self.config.finetuned_tokenizer_path)
        else:
            if self.rank == 0:
                print("pre_trained_tokenizer=False, randomly initializing Tokenizer architecture for Predictor training")
            import json
            cfg_path = os.path.join(self.config.pretrained_tokenizer_path, 'config.json')
            with open(cfg_path, 'r') as f:
                arch = json.load(f)
            tokenizer = KronosTokenizer(
                d_in=arch.get('d_in', 6),
                d_model=arch.get('d_model', 256),
                n_heads=arch.get('n_heads', 4),
                ff_dim=arch.get('ff_dim', 512),
                n_enc_layers=arch.get('n_enc_layers', 4),
                n_dec_layers=arch.get('n_dec_layers', 4),
                ffn_dropout_p=arch.get('ffn_dropout_p', 0.0),
                attn_dropout_p=arch.get('attn_dropout_p', 0.0),
                resid_dropout_p=arch.get('resid_dropout_p', 0.0),
                s1_bits=arch.get('s1_bits', 10),
                s2_bits=arch.get('s2_bits', 10),
                beta=arch.get('beta', 0.05),
                gamma0=arch.get('gamma0', 1.0),
                gamma=arch.get('gamma', 1.1),
                zeta=arch.get('zeta', 0.05),
                group_size=arch.get('group_size', 4)
            )
        tokenizer = tokenizer.to(self.device)
        
        if getattr(self.config, 'pre_trained_predictor', True):
            logger.info("Loading pretrained predictor...")
            if self.rank == 0:
                print("Loading pretrained predictor...")
            model = Kronos.from_pretrained(self.config.pretrained_predictor_path)
        else:
            if self.rank == 0:
                print("pre_trained_predictor=False, randomly initializing Predictor architecture")
            import json
            cfg_path = os.path.join(self.config.pretrained_predictor_path, 'config.json')
            with open(cfg_path, 'r') as f:
                arch = json.load(f)
            print("model_config: ", arch)
            model = Kronos(
                s1_bits=arch.get('s1_bits', 10),
                s2_bits=arch.get('s2_bits', 10),
                n_layers=arch.get('n_layers', 12),
                d_model=arch.get('d_model', 832),
                n_heads=arch.get('n_heads', 16),
                ff_dim=arch.get('ff_dim', 2048),
                ffn_dropout_p=arch.get('ffn_dropout_p', 0.2),
                attn_dropout_p=arch.get('attn_dropout_p', 0.0),
                resid_dropout_p=arch.get('resid_dropout_p', 0.2),
                token_dropout_p=arch.get('token_dropout_p', 0.0),
                learn_te=arch.get('learn_te', True)
            )
        model = model.to(self.device)
        
        model_size = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {model_size:,}")
        if self.rank == 0:
            print(f"Model parameters: {model_size:,}")
        
        logger.info("=== Training Configuration ===")
        logger.info(f"Data path: {self.config.data_path}")
        logger.info(f"Lookback window: {self.config.lookback_window}")
        logger.info(f"Predict window: {self.config.predict_window}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.predictor_learning_rate}")
        logger.info(f"Training epochs: {self.config.basemodel_epochs}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Tokenizer path: {self.config.finetuned_tokenizer_path}")
        logger.info(f"Pretrained model path: {self.config.pretrained_predictor_path}")
        
        logger.info("Starting fine-tuning training...")
        if self.rank == 0:
            print("Starting fine-tuning training...")
        start_time = time.time()
        best_val_loss = train_model(
            model,
            tokenizer,
            self.device,
            self.config,
            self.config.basemodel_save_path,
            logger,
        )
        training_time = time.time() - start_time
        
        final_msg = f"Basemodel training completed! Best validation loss: {best_val_loss:.4f}\nTraining time: {training_time/60:.2f} minutes\nModel saved to: {self.config.basemodel_save_path}"
        logger.info(final_msg)
        if self.rank == 0:
            print(f"\n{final_msg}")
        
        return True
    
    def run_training(self):
        if self.rank == 0:
            print("Starting Kronos model sequential fine-tuning training")
            print(f"Experiment name: {self.config.experiment_name}")
            print(f"Experiment description: {self.config.experiment_description}")
        
        self._setup_distributed()
        
        self._create_directories()
        
        tokenizer_exists, basemodel_exists = self._check_existing_models()
        
        total_start_time = time.time()
        
        try:
            if self.config.train_tokenizer:
                success = self.train_tokenizer_phase()
                if not success:
                    print("Tokenizer training failed, terminating training")
                    return False
            else:
                print("Skipping Tokenizer training phase")
            
            if self.config.train_basemodel:
                success = self.train_basemodel_phase()
                if not success:
                    print("Basemodel training failed, terminating training")
                    return False
            else:
                print("Skipping Basemodel training phase")
            
            total_time = time.time() - total_start_time
            
            if self.rank == 0:
                print("\n" + "="*60)
                print("Training completed!")
                print("="*60)
                print(f"Total training time: {total_time/60:.2f} minutes")
                print(f"Tokenizer model: {self.config.tokenizer_best_model_path}")
                print(f"Basemodel model: {self.config.basemodel_best_model_path}")
                print("="*60)
            
            return True
            
        except Exception as e:
            if self.rank == 0:
                print(f"Error occurred during training: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            pass


def main():
    parser = argparse.ArgumentParser(description='Kronos Model Sequential Fine-tuning Training')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Configuration file path (default: config.yaml)')
    parser.add_argument('--skip-tokenizer', action='store_true', 
                       help='Skip tokenizer training phase')
    parser.add_argument('--skip-basemodel', action='store_true', 
                       help='Skip basemodel training phase')
    parser.add_argument('--skip-existing', action='store_true', 
                       help='Skip training for existing models')
    
    args = parser.parse_args()
    
    trainer = SequentialTrainer(args.config)
    
    if args.skip_tokenizer:
        trainer.config.train_tokenizer = False
    if args.skip_basemodel:
        trainer.config.train_basemodel = False
    if args.skip_existing:
        trainer.config.skip_existing = True
    
    success = trainer.run_training()
    
    if success:
        print("Training completed successfully!")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        sys.exit(0)
    else:
        print("Training failed!")
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
                dist.destroy_process_group()
            except Exception:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
