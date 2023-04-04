# Training Large models using DeepSpeed

Welcome to the DeepSpeed training tutorial! DeepSpeed is a deep learning optimization library that enables the training of large-scale deep learning models with minimal code changes. With DeepSpeed, you can train models with billions of parameters on a single GPU.

In this example, we will show you how to train a large model using DeepSpeed on NimbleBox. To get started, follow these steps:

- Run the following command to upload the training job, it will allocate the necessary resources for your job, including CPU, memory, disk space, and GPU. You can adjust the resources as per your requirements.

```bash
nbx jobs upload train:main --trigger --id '<your-id>' \
  --resource_cpu="600m" \
  --resource_memory="600Mi" \
  --resource_disk_size="10Gi" \
  --resource_gpu="nvidia-tesla-k80" \
  --resource_gpu_count="1"
```

- The code in `train:main` will train your model using DeepSpeed. The code is optimized for distributed training, which enables you to train the model on multiple GPUs which you can configure using `--resource_gpu_count`. The code will also save the model weights to disk after training.

DeepSpeed is a powerful tool for training large models with minimal code changes. With NimbleBox, you can easily allocate the necessary resources and train your models efficiently.

Thank you for choosing NimbleBox for your machine learning needs!
