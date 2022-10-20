import torch
from time import time
from torch import autocast
from diffusers import StableDiffusionPipeline

from nbox import operator
from nbox.utils import b64encode

HF_TOKEN = "<your-hf-token-here>"

@operator()
class StableDiff:
  def __init__(self):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading model on:", self.device)
    self.pipe = StableDiffusionPipeline.from_pretrained(
      "CompVis/stable-diffusion-v1-4",
      use_auth_token = HF_TOKEN,
      cache_dir = "./",
    ).to(self.device)

  def _generate(
    self,
    text: str,
    height = 512,
    width = 512,
    num_inference_steps = 100,
    guidance_scale = 20,
    eta = 0.4,
  ):
    with autocast(self.device):
      st = time()
      out = self.pipe(
        prompt = text,
        height = height,
        width = width,
        num_inference_steps = num_inference_steps,
        guidance_scale = guidance_scale,
        eta = eta
      )
      return out, time() - st

  def generate(
    self,
    text: str,
    height = 512,
    width = 512,
    num_inference_steps = 100,
    guidance_scale = 20,
    eta = 0.4,
  ):
    """This endpoint is used to generate a single image, these queries have highest priority"""
    out, time_taken = self._generate(
      text = text,
      height = height,
      width = width,
      num_inference_steps = num_inference_steps,
      guidance_scale = guidance_scale,
      eta = eta,
    )
    return {
      "nsfw_content_detected": out["nsfw_content_detected"],
      "images": [b64encode(i.tobytes()) for i in out["images"]],
      "time_taken": time_taken,
    }
