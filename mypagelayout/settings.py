import os
from typing import Callable, Dict, Optional

import torch
from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings
from pathlib import Path
from platformdirs import user_cache_dir

class Settings(BaseSettings):
    # General
    TORCH_DEVICE: Optional[str] = None
    DISABLE_TQDM: bool = False
    S3_BASE_URL: str = "https://models.datalab.to" # Default from original
    PARALLEL_DOWNLOAD_WORKERS: int = 10 # Default from original
    MODEL_CACHE_DIR: str = str(Path(user_cache_dir("datalab")) / "models") # Default from original
    LOGLEVEL: str = "INFO"  # Default from original

    # Layout specific
    LAYOUT_MODEL_CHECKPOINT: str = "s3://layout/2025_02_18" # Default from original
    LAYOUT_IMAGE_SIZE: Dict = {"height": 768, "width": 768} # Default from original
    LAYOUT_SLICE_MIN: Dict = {"height": 1500, "width": 1500} # Default from original
    LAYOUT_SLICE_SIZE: Dict = {"height": 1200, "width": 1200} # Default from original
    LAYOUT_BATCH_SIZE: Optional[int] = None
    LAYOUT_MAX_BOXES: int = 100 # Default from original
    COMPILE_LAYOUT: bool = False
    RECOGNITION_PAD_VALUE: int = 255 # Default from original, used in processor

    # Dependencies for computed fields
    COMPILE_ALL: bool = False

    @computed_field
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        try:
            import torch_xla
            if len(torch_xla.devices()) > 0:
                return "xla"
        except Exception:
            pass

        return "cpu"

    @computed_field
    def MODEL_DTYPE(self) -> torch.dtype:
        if self.TORCH_DEVICE_MODEL == "cpu":
            return torch.float32
        if self.TORCH_DEVICE_MODEL == "xla":
            return torch.bfloat16
        return torch.float16

    @computed_field
    def LAYOUT_STATIC_CACHE(self) -> bool:
        return (
            self.COMPILE_ALL or self.COMPILE_LAYOUT or self.TORCH_DEVICE_MODEL == "xla"
        )

    @computed_field
    def INFERENCE_MODE(self) -> Callable:
        if self.TORCH_DEVICE_MODEL == "xla":
            return torch.no_grad
        return torch.inference_mode

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"

settings = Settings()
