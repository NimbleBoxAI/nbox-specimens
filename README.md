<a href="https://nimblebox.ai/" target="_blank"><img src="./assets/built_at_nbx.svg" align="right"></a>

# nbox-specimens

This repository contains follow along material and examples of using [`nbox`](https://github.com/NimbleBoxAI/nbox). In case you do not have `nbox` installed:

```bash
pip install nbox

# call on CLI once to authenticate
nbx why
```

## 🍇 How to use

- [**End to End**] [The stable-diffusion AI startup](./dreambooth-stable-diff/): [**GPU**] Train a Stable Diffusion model and serve it.

## 🌳 Projects

The projects here are used to demonstrate how to run different kinds of workloads and not the content of the operation.

- [**Jobs**] [Sklearn Tests](./sklearn_tests/): Train a simple kmeans model and store the resulting chart in a Relic.
- [**Jobs + LMAO**] [lmao](./lmao/): Use NBX-Monitoring projects to track, train and store your model giving a full end to end MLOps pipeline.
- [**Serving**] [Pose Model](./posemodel/): Deploy a `mediapipe` pose detection model on a kubernetes cluster (w/o YAML 😛)
- [**Serving**] [FastAPI](./fastapi_serving/): Deploy a FastAPI server on a NBX-Serving
- [**Jobs**] [DeepSpeed](./deepspeed): [**GPU**] Train a MobileViT using [DeepSpeed](https://www.deepspeed.ai/)
- [**Jobs + Deploy**] [Compute Fabric](./compute_fabric): Automate the compute via scripts and super power your local developement
