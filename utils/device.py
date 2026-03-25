import os
import logging
import torch

logger = logging.getLogger(__name__)

def get_optimal_device() -> torch.device:
    """
    Automatically selects the best hardware acceleration device:
    1. NVIDIA/AMD (CUDA interface, ROCm is also mapped to CUDA in PyTorch)
    2. MAC (MPS)
    3. CPU (Fallback)
    """
    if torch.cuda.is_available():
        # ROCm users will also hit this path as PyTorch maps ROCm to cuda
        return torch.device("cuda")
    
    # Check for Apple Silicon MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple Silicon (MPS) acceleration detected.")
        return torch.device("mps")
    
    logger.warning("No hardware acceleration (CUDA/MPS) found. Falling back to CPU. Performance will be degraded.")
    return torch.device("cpu")

def setup_hardware_env():
    """
    Environment-specific setup to avoid common library conflicts.
    """
    # Fix for double OpenMP initialization error on macOS
    if os.sys.platform == "darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        logger.info("Applied macOS OpenMP conflict workaround (KMP_DUPLICATE_LIB_OK=TRUE)")

def get_device_name(device: torch.device) -> str:
    """Returns a human-readable name for the device."""
    if device.type == "cuda":
        return f"NVIDIA/AMD GPU ({torch.cuda.get_device_name(0)})"
    elif device.type == "mps":
        return "Apple Silicon GPU (MPS)"
    return "Generic CPU"
