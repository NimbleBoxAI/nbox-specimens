import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

import json
from subprocess import Popen
from transformers import TrainingArguments, Trainer
from transformers import DefaultDataCollator
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification
from datasets import load_dataset
from PIL import Image
import requests
import numpy as np
import torch
import evaluate


def main():
  def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = feature_extractor(images=[i.convert("RGB") for i in example_batch["image"]], return_tensors="pt")["pixel_values"]
    del example_batch["image"]
    return example_batch

  def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

  # load model and all that
  feature_extractor = MobileViTFeatureExtractor.from_pretrained("apple/mobilevit-small")
  model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
  setattr(model.config, "hidden_size", 240)
  dataset = load_dataset("Maysee/tiny-imagenet", split="train[:5000]").train_test_split(test_size=0.2)
  train_set = dataset["train"]
  valid_set = dataset["test"]
  print(train_set)
  print(valid_set)
  train_set.set_transform(preprocess_train)
  valid_set.set_transform(preprocess_train)
  metric = evaluate.load("accuracy")

  # write the config for deepspeed
  with open("ds_config_zero3.json", "w") as f:
    f.write(json.dumps({
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
      "stage": 2,
      "offload_optimizer": {
        "device": "cpu",
        "pin_memory": True
      },
      "allgather_partitions": True,
      "allgather_bucket_size": 5e8,
      "overlap_comm": True,
      "reduce_scatter": True,
      "reduce_bucket_size": 5e8,
      "contiguous_gradients": True
    },
    "train_batch_size": "auto"
  }))

  Popen("ds_report", shell = True).wait()
  training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="steps",
    remove_unused_columns=False,
    report_to="all",
    num_train_epochs=4,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    save_total_limit=2,
    deepspeed="ds_config_zero3.json"
  )
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_set,
      eval_dataset=valid_set,
      compute_metrics=compute_metrics,
  )
  trainer.train()
