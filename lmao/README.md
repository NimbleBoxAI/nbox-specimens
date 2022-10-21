# Full Training using LMAO

> LMAO is a new service being released in private Beta by NimbleBox. If you do not see "Monitoring" tab reach out to us via Intercom.

How can you use the NimbleBox LMAO for training, logging and storing of artifacts of any ML model. Here we are going to train a really simple single layer neural network which will log all the metrics, store artifacts in the relevant location and optionally deploy the model as an API endpoint.

- In case you do not have a sample dataset in "nbx_core" relic upload using `python3 upload.py`, this is a one time thing.

To upload run your code:

```bash
nbx lmao upload trainer:train_model <job_id> \
  --trigger \           # not just upload but also trigger the run
  --n 20 \              # you can pass function kwargs through CLI
  --resource_cpu="120m" # give resources that you want
```
