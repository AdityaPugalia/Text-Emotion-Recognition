import torch


def get_best_device() -> torch.device:
    """
    Gets the best available device for training.

    Order of preference:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon GPU)
    3. CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU (CUDA) for training.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using GPU (MPS) for training.")
    else:
        device = torch.device("cpu")
        print("Using CPU for training.")
    return device
