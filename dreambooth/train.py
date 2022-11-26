import math
import time
import torch
import itertools
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from functools import partial
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from dreambooth_dataset import collate_fn, DreamBoothDataset
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline


def main(
  pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4",
  tokenizer_name = None, 
  logging_dir = 'logs',
  instance_prompt = 'a photo of sks man',
  revision = None,
  gradient_accumulation_steps = 1,
  mixed_precision = 'no',
  train_text_encoder = True,
  seed = None,
  learning_rate = 2e-6,
  scale_lr = False,
  train_batch_size = 1,
  adam_beta1 = 0.9,
  adam_beta2 = 0.999,
  adam_weight_decay = 1e-2,
  adam_epsilon = 1e-08, 
  size = 512,
  center_crop = False,
  max_train_steps = 5,
  num_train_epochs = None, 
  lr_scheduler_type = 'constant',
  lr_warmup_steps = 0,
  with_prior_preservation = False,
  max_grad_norm = 1.0
):
  start = time.time()

  instance_data_root = f'./sample_pics',
  output_dir = f'./outputs/',

  config = {
    'pretrained_model_name_or_path': pretrained_model_name_or_path,
    'tokenizer_name': tokenizer_name,
    'instance_data_root': instance_data_root,
    'logging_dir': logging_dir,
    'output_dir': output_dir,
    'instance_prompt': instance_prompt,
    'revision': revision,
    'gradient_accumulation_steps': gradient_accumulation_steps,
    'mixed_precision': mixed_precision,
    'train_text_encoder': train_text_encoder,
    'seed': seed,
    'learning_rate': learning_rate,
    'scale_lr': scale_lr,
    'train_batch_size': train_batch_size,
    'adam_beta1': adam_beta1,
    'adam_beta2': adam_beta2,
    'adam_weight_decay': adam_weight_decay,
    'adam_epsilon': adam_epsilon,
    'size': size,
    'center_crop': center_crop,
    'train_batch_size': train_batch_size,
    'max_train_steps': max_train_steps,
    'num_train_epochs': num_train_epochs,
    'lr_scheduler_type': lr_scheduler_type,
    'lr_warmup_steps': lr_warmup_steps,
    'with_prior_preservation': with_prior_preservation,
    'max_grad_norm': max_grad_norm,
  }

  # Step 1: Initialize Accelerator
  accelerator = Accelerator(
    gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision=mixed_precision,
    log_with="tensorboard", logging_dir=logging_dir
  )

  # Step 2: Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate 
  if train_text_encoder and gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
      raise ValueError(
        "Gradient accumulation is not supported when training the text encoder in distributed training. "
        "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
      )
  
  # Step 3: Set seed
  if seed is not None:
    set_seed(seed)

  # Step 4: Define tokenizer
  if tokenizer_name:
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name, revision=revision)
    print(f'Loaded tokenizer from tokenizer_name: {tokenizer_name}')
  else:
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", revision=revision)
    print(f'Loaded tokenizer from pretrained_model_name_or_path: {pretrained_model_name_or_path}')
  
  # Step 5: Load models and create wrapper for stable diffusion
  # Models: text_encoder, vae, unet
  text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder='text_encoder', revision=revision)
  vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder='vae', revision=revision)
  unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder='unet', revision=revision)
  noise_scheduler = DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder='scheduler')

  # Step 6: scale_lr
  if scale_lr:
    learning_rate = learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes

  # Step 7: Choose parameters to optimize and define optimizer
  vae.requires_grad_(False)
  if not train_text_encoder:
    text_encoder.requires_grad_(False)    
  params_to_optimize = (itertools.chain(unet.parameters(), text_encoder.parameters()) if train_text_encoder else unet.parameters())

  optimizer_class = torch.optim.AdamW
  optimizer = optimizer_class(
    params_to_optimize, 
    lr=learning_rate,
    betas=(adam_beta1, adam_beta2),
    weight_decay=adam_weight_decay,
    eps=adam_epsilon
  )

  # Step 8: Define dataset and dataloader
  train_dataset = DreamBoothDataset(
    instance_data_root=instance_data_root[0],
    instance_prompt=instance_prompt,
    tokenizer=tokenizer,
    size=size,
    center_crop=center_crop
  )

  collate_fn = partial(collate_fn, tokenizer=tokenizer)
  train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=1
  )

  # Step 9: Math around the number of training steps
  overrode_max_train_steps = False
  num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
  if max_train_steps is None:
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

  # Step 10a: Define the scheduler
  lr_scheduler = get_scheduler(
    lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
    num_training_steps=max_train_steps * gradient_accumulation_steps
  )

  # Step 10b: Visualize the scheduler
  learning_rates = []
  for i in range(max_train_steps * gradient_accumulation_steps):
    optimizer.step()
    lr_scheduler.step()
    learning_rates.append(optimizer.param_groups[0]["lr"])

  plt.figure(figsize=(4,1.5))
  _ = plt.plot(range(max_train_steps*gradient_accumulation_steps), learning_rates)
  _ = plt.title('Learning rate scheduler')
  _ = plt.xlabel('Training iterations')
  _ = plt.ylabel('Learning rate')

  plt.savefig('lr_scheduler.png') # We will push it Relics Later

  # Step 11: Prepare accelerator
  if train_text_encoder:
      unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler
      )
  else:
      unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
      )
  
  # Step 12: Mixed precision training
  weight_dtype = torch.float32
  if mixed_precision == 'fp16':
    weight_dtype = torch.float16
  elif mixed_precision == 'bf16':
    weight_dtype = torch.bfloat16

  # Step 13: Move text_encoder and vae to GPU
  # For mixed precision training we cast the vae weights to half-precision as it is used only for inference
  # If we are not training the text_encoder we cast its weights to half-precision as well
  vae.to(accelerator.device, dtype=weight_dtype)
  if not train_text_encoder:
    text_encoder.to(accelerator.device, dtype=weight_dtype)

  # Step 14: We need to recalculate the total training steps as the size of the training dataloader may have changed
  num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
  if overrode_max_train_steps:
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
  num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

  # Step 15: Initializing the trackers and storing our configuration
  if accelerator.is_main_process:
      accelerator.init_trackers('dreambooth', config=config)

  timesteps = 15
  f, axarr = plt.subplots(1,2)
  _ = axarr[0].imshow(transforms.ToPILImage()(train_dataset[0]['instance_images']))
  _ = axarr[1].imshow(transforms.ToPILImage()(
    noise_scheduler.add_noise(
      train_dataset[0]['instance_images'],
      torch.randn_like(train_dataset[0]['instance_images']),
      torch.Tensor([timesteps]).long()
    )
  ))

  # Again push to Relics Later

  # Step 16: Training logs
  total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

  print(f'***** Starting training *****')
  print(f'Number of training examples: {len(train_dataset)}')
  print(f'Number of batches in each epoch: {len(train_dataloader)}')
  print(f'Number of training epochs: {num_train_epochs}')
  print(f'Batch size per device: {train_batch_size}')
  print(f'Total train batch size with parallel, distributed and accumulation: {total_batch_size}')
  print(f'Gradient accumulation steps: {gradient_accumulation_steps}')
  print(f'Max train steps: {max_train_steps}')

  # Step 17: Start training
  progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
  progress_bar.set_description("Steps")
  global_step = 0

  for epoch in range(num_train_epochs):
    unet.train()
    if train_text_encoder:
      text_encoder.train()
    for step, batch in enumerate(train_dataloader):
      with accelerator.accumulate(unet):
        # Step 17a: Convert images to latent space
        # Input: batch['pixel_values'] shape: torch.Size([1, 3, 512, 512])
        # Output: latents -> AutoencoderKLOutput(latent_dist=)
        latents = vae.encode(batch['pixel_values'].to(dtype=weight_dtype))
        # Output: latents shape: torch.Size([1, 4, 64, 64])
        latents = latents.latent_dist.sample()
        latents = latents * 0.18215
        # Step 17b: Sample noise that we will add to the latents
        # Output: noise shape: torch.Size([1, 4, 64, 64])
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Step 17c: Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        # Step 17d: Forward Diffusion: Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        # Step 17e: Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(batch['input_ids']).last_hidden_state
        # Step 17f: Predict the noise residual
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        # Step 17g: Compute loss
        if with_prior_preservation:
            pass
        else:
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        # Step 17h: Run backpropagation
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            params_to_clip = (
                itertools.chain(unet.parameters(), text_encoder.parameters()) if train_text_encoder
                else unet.parameters()
            )
            accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
      if accelerator.sync_gradients:
          progress_bar.update(1)
          global_step += 1
      logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
      progress_bar.set_postfix(**logs)
      accelerator.log(logs, step=global_step)
      if global_step >= max_train_steps:
          break

  if accelerator.is_main_process:
    pipeline = StableDiffusionPipeline.from_pretrained(
      pretrained_model_name_or_path,
      unet=accelerator.unwrap_model(unet),
      text_encoder=accelerator.unwrap_model(text_encoder),
      revision=revision
    )
    pipeline.save_pretrained(output_dir)
  accelerator.end_training()

  end = time.time()
  print(end - start)
