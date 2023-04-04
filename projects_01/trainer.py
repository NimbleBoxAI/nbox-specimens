import os
from time import sleep
from tqdm import trange
from nbox import Project, operator, logger, lo

import torch
from torch.nn import functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

from model import get_data, get_model
from const import project_id

@operator()
def train_model(
  n_steps: int = 50,
  batch_size: int = 64,
  lr: float = 0.01,
  checkpoint_every: int = 25,
):
  # create a project, if this is running on pod then all the initializations are already done
  # it already knows the project id and experiment id
  p = Project(project_id)
  relic = p.get_relic()
  exp_tracker = p.get_exp_tracker(
    metadata = {
      "lr": lr
    }  
  )

  # copy the data from the relic and structure to tensors
  if not os.path.exists("data.csv"):
    relic.get_from("data.csv", f"datasets/data_1000_5.csv")

  df = pd.read_csv("data.csv")
  data = np.concatenate([
    df["a"].values.reshape(-1, 1),
    df["b"].values.reshape(-1, 1),
    df["c"].values.reshape(-1, 1),
    df["d"].values.reshape(-1, 1),
    df["e"].values.reshape(-1, 1)],
    axis = 1
  )
  labels = df["label"].values
  data = torch.tensor(data).float()
  labels = torch.tensor(labels).long()

  # train your model
  model = get_model(5, 5)
  adam = torch.optim.Adam(model.parameters(), lr = lr)
  for i, batch in zip(
    trange(n_steps),
    get_data(data, labels, batch_size, repeat = True)
  ):
    adam.zero_grad()
    logits = model(batch["input"])
    loss = F.cross_entropy(logits, batch["labels"])
    loss.backward()
    adam.step()
    y_pred = logits.argmax(dim = 1)
    y_true = batch["labels"]
    acc = (y_pred == y_true).float().mean().item()
    record = {
      "loss": loss.item(),
      "accuracy": acc,
      "kappa": float(cohen_kappa_score(y_pred, y_true)),
      "mcc": float(matthews_corrcoef(y_pred, y_true)),
      "step": i,
    }
    # logger.info(lo("Step", i, ":", **record)) # pretty logging
    exp_tracker.log(record, step = i)

    # save file and automatically sync checkpoints
    if checkpoint_every and (i+1) % checkpoint_every == 0:
      folder_name = f"model_{i:03d}"
      logger.info(lo("Saving model at", folder_name, "...")) # pretty logging
      os.makedirs(folder_name, exist_ok = True)
      torch.save(model.state_dict(), f"{folder_name}/model.pt")
      exp_tracker.save_file(folder_name)

    sleep(1) # simulate training time

  # end the run
  exp_tracker.end()

if __name__ == "__main__":
  train_model()