# Training Large models using DeepSpeed

DeepSpeed is a deep learning optimization library that is designed to enable training of large-scale deep learning models with minimal code changes.

All you have to do is:

```
nbx jobs upload train:main --trigger --id '<your-id>' --resource_cpu="600m" --resource_memory="600Mi" --resource_disk_size="10Gi" --resource_gpu="nvidia-tesla-k80" --resource_gpu_count="1"
```
