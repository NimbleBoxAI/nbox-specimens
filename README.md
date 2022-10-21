<a href="https://nimblebox.ai/" target="_blank"><img src="./assets/built_at_nbx.svg" align="right"></a>

# nbox-specimens

This repository contains follow along material and examples of using [`nbox`](https://github.com/NimbleBoxAI/nbox).

## 🌳 Projects

The projects here are used to demonstrate how to run different kinds of workloads and not the content of the operation.

- [**Jobs**] [Sklearn Tests](./sklearn_tests/): Train a simple kmeans model and store the resulting chart in a Relic.
- [**Jobs + LMAO**] [lmao](./lmao/): Use the upcoming LMAO projects to track, train and store your model giving a full end to end MLOps pipeline.

## 🚧 In Progress

- [**Deploy**] [Stable Diffusion](./stable_diff/): Deploy a Stable Diffusion model

### Deprecated Examples

These examples will be migrated to the new format for deploying batch processes (serverless) and API endpoints.

- [**Jobs & Deploy**] [A Job that deploys itself](./deploy_itself/)
- [**Jobs**] [Credit Card Fraud Model](./jobs_credit_card_fraud/): This will download a ~80MB csv file, preprocess and convert to a dataframe, do grid search for best model and securely transfer it back to a NBX-Build instance.
- [**Jobs**] [GPU Jobs using `faiss`](./jobs_faiss_gpu/): Creates a GPU NBX-Job and runs the tests from [facebookresearch/faiss](https://github.com/facebookresearch/faiss)

- [**Jobs**] [gperc-insanity](./gperc-insanity/)
