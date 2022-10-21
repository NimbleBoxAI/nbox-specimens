# Stable Diffusion on NimbleBox

Stable Diffusion is cool, how can you use it? Simple example on how to deploy a Stable diffusion model on the NimbleBox platform. It's quite simple actually, write a simple class as if you are going to import it locally and use it and add a `@operator()` decorator on top of it. This will convert a class into an `nbox.Operator` object and then we can use the powerful NimbleBox backend.

- Install nbox `pip install nbox>=0.10.6`
- Get your token from [hf.co](https://huggingface.co/settings/tokens) and paste it in `HF_TOKEN` in `model.py`
- Upload the model and run it under the deployment name `stable_diff_api`

```
nbx serve upload model:StableDiff 'stable_diff'
```
