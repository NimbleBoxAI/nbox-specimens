import os
from nbox import operator
from nbox.projects import Project

import torch
from torch.nn import functional as F
import numpy as np
import pandas as pd

from model import get_data, get_model

@operator()
def train_model(n_steps: int = 20, batch_size: int = 64, lr: float = 0.01):
  # create a project, if this is running on pod then all the initializations are already done
  # it already knows the project id and experiment id
  p = Project()
  relic = p.get_relic()
  exp_tracker = p.get_exp_tracker()

  # copy the data from the relic and structure to tensors
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
    range(n_steps),
    get_data(data, labels, batch_size, repeat = True)
  ):
    adam.zero_grad()
    logits = model(batch["input"])
    loss = F.cross_entropy(logits, batch["labels"])
    loss.backward()
    adam.step()
    print(f"Step {i:03d} loss is {loss.item():.3f}")
    exp_tracker.log(
      {
        "loss": loss.item(),
        "accuracy": (logits.argmax(dim = 1) == batch["labels"]).float().mean().item(),
      },
      step = i,
    )

    # save file and automatically sync checkpoints
    if i % 5 == 0:
      folder_name = f"model_{i:03d}"
      os.makedirs(folder_name, exist_ok = True)
      torch.save(model.state_dict(), f"{folder_name}/model.pt")
      exp_tracker.save_file(folder_name)

  # end the run
  exp_tracker.end()
