<a href="https://nimblebox.ai/" target="_blank"><img src="./assets/built_at_nbx.svg" align="right"></a>

# nbox-specimens

This repository contains follow along material and examples of using [`nbox`](https://github.com/NimbleBoxAI/nbox). In case you do not have `nbox` installed:

```bash
pip install nbox

# call on CLI once to authenticate
nbx why
```

> If any of the examples break, please try updating `nbox` to the latest version.

## 🍇 How to use NBX-Projects

- [projects_01](./projects_01/): Use the NimbleBox projects to train a pytorch model, visualise the logs and store the relevant artifacts.
- [The stable-diffusion AI startup](./dreambooth-stable-diff/): [**GPU**] Train a Stable Diffusion model and serve it.
- [Mastering Blackjack with Q-learning](./blackjack/): Train a Q-learning agent to play Blackjack.

## 🌳 Service-Specific

The projects here are used to demonstrate how to run different kinds of workloads and not the content of the operation.

- [**Jobs**] [Sklearn Tests](./sklearn_tests/): Train a simple kmeans model and store the resulting chart in a Relic.
- [**Serving**] [Pose Model](./posemodel/): Deploy a `mediapipe` pose detection model on a kubernetes cluster (w/o YAML 😛)
- [**Serving**] [FastAPI](./fastapi_serving/): Deploy a FastAPI server on a NBX-Serving
- [**Jobs**] [DeepSpeed](./deepspeed): [**GPU**] Train a MobileViT using [DeepSpeed](https://www.deepspeed.ai/)

## 🌌 More ambitious projects

These are projects are for the ambitious ones who want to try out more advanced features of NBX.

- [Compute Fabric](./compute_fabric): Attach powerful cloud compute and super power your local developement
