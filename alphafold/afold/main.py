import os

from nbox import logger

# Confirm accelerator configuration
import jax

def print_device_info()
  if jax.local_devices()[0].platform == "tpu":
    raise RuntimeError(
        "TPU runtime not supported. Please configure GPU acceleration on the VM."
    )
  elif jax.local_devices()[0].platform == "cpu":
    logger.info(
        "CPU-only runtime is not recommended, because prediction execution will be slow. For better performance, consider GPU acceleration on the VM."
    )
  else:
    logger.info(f"Running with {jax.local_devices()[0].device_kind} GPU")

  # Make sure all necessary environment variables are set.
  

  os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
  os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "2.0"
