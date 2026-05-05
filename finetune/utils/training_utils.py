import os
import random
import datetime
import numpy as np
import torch
import torch.distributed as dist


def setup_ddp():
    """
    Initializes the distributed data parallel environment.

    This function relies on environment variables set by `torchrun` or a similar
    launcher. It initializes the process group and sets the CUDA device for the
    current process.

    Returns:
        tuple: A tuple containing (rank, world_size, local_rank).
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available.")

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(
        f"[DDP Setup] Global Rank: {rank}/{world_size}, "
        f"Local Rank (GPU): {local_rank} on device {torch.cuda.current_device()}"
    )
    return rank, world_size, local_rank


def cleanup_ddp():
    """Cleans up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int = 0):
    """
    Sets the random seed for reproducibility across all relevant libraries.

    Args:
        seed (int): The base seed value.
        rank (int): The process rank, used to ensure different processes have
                    different seeds, which can be important for data loading.
    """
    actual_seed = seed + rank
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
        # The two lines below can impact performance, so they are often
        # reserved for final experiments where reproducibility is critical.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_size(model: torch.nn.Module) -> str:
    """
    Calculates the number of trainable parameters in a PyTorch model and returns
    it as a human-readable string.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        str: A string representing the model size (e.g., "175.0B", "7.1M", "50.5K").
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if total_params >= 1e9:
        return f"{total_params / 1e9:.1f}B"  # Billions
    elif total_params >= 1e6:
        return f"{total_params / 1e6:.1f}M"  # Millions
    else:
        return f"{total_params / 1e3:.1f}K"  # Thousands


def reduce_tensor(tensor: torch.Tensor, world_size: int, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    Reduces a tensor's value across all processes in a distributed setup.

    Args:
        tensor (torch.Tensor): The tensor to be reduced.
        world_size (int): The total number of processes.
        op (dist.ReduceOp, optional): The reduction operation (SUM, AVG, etc.).
                                      Defaults to dist.ReduceOp.SUM.

    Returns:
        torch.Tensor: The reduced tensor, which will be identical on all processes.
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=op)
    # Note: `dist.ReduceOp.AVG` is available in newer torch versions.
    # For compatibility, manual division is sometimes used after a SUM.
    if op == dist.ReduceOp.AVG:
        rt /= world_size
    return rt


def format_time(seconds: float) -> str:
    """
    Formats a duration in seconds into a human-readable H:M:S string.

    Args:
        seconds (float): The total seconds.

    Returns:
        str: The formatted time string (e.g., "0:15:32").
    """
    return str(datetime.timedelta(seconds=int(seconds)))



def resolve_amp_dtype(amp_dtype):
    """
    Resolves the configured AMP dtype string into the arguments expected by
    `torch.autocast`.

    Currently only "bfloat16" is supported. Passing ``None`` disables
    mixed-precision training; the returned dtype is then irrelevant and
    autocast becomes a no-op.

    Args:
        amp_dtype (str | None): "bfloat16" or None.

    Returns:
        tuple[torch.dtype, bool]: (dtype, enabled) suitable for
        ``torch.autocast(device_type=..., dtype=dtype, enabled=enabled)``.

    Raises:
        ValueError: If ``amp_dtype`` is set to an unsupported value.
    """
    if amp_dtype is None:
        return torch.float32, False
    if amp_dtype == "bfloat16":
        return torch.bfloat16, True
    raise ValueError(
        f"Unsupported amp_dtype {amp_dtype!r}; expected 'bfloat16' or None."
    )
