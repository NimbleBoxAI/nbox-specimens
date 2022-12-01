from nbox import Operator, Relics, Lmao, operator, Resource
from nbox import __version__

import torch
import numpy as np
import pandas as pd
from tqdm import trange
from time import sleep


class Estimator(torch.nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.hidden = torch.nn.Linear(5, 5)

  def forward(self, x):
    return self.hidden(x)


# object that you want to serve over the internet
class TorchInferenceModel(Operator):
  def __remote_init__(self):
    self.model = Estimator()
    get_relic().get_from("model.pt", f"checkpoints/model.pt")
    self.model.load_state_dict(torch.load("model.pt"))

  def forward(self, x):
    if isinstance(x[0], list):
      x = torch.tensor(x)
    tensor = torch.tensor(x).float()
    return self.model(tensor).tolist()


@operator()
def train_model(n: int = 10, lr: float = 0.01, deploy: bool = False):
  # say you have a file(s) in `nbx_core` relic at datasets/data_1000_5.csv
  # you will download it at "data.csv"
  get_relic("nbx_core").get_from("data.csv", f"datasets/data_1000_5.csv")
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

  # train your model, lmao client already knows the project_id during deployment
  model = Estimator()
  adam = torch.optim.Adam(model.parameters(), lr = lr)
  lmao = Lmao(metadata = {"model": "demo_model", "training_steps": n})
  for i in trange(n):
    adam.zero_grad()
    tensor = torch.tensor(data).float()
    out = model(tensor)
    loss = torch.nn.functional.cross_entropy(out, torch.tensor(labels))
    loss.backward()
    adam.step()
    lmao.log({"loss": loss.item()}, i)
    sleep(1.5)

  # save file
  torch.save(model.state_dict(), "model.pt")
  lmao.save_file("model.pt")

  # say you want to store your files on another relic at some other location
  # get_relic("e2e_demo").put_to("model.pt", f"checkpoints/model.pt")

  # end the run
  lmao.end()

  if deploy:
    model = TorchInferenceModel()
    model.deploy("serving", resource = Resource(
      cpu = "100m",
      memory = "3Gi",
      max_retries = 2,
      disk_size = "5Gi"
    ))


def get_relic(name):
  relic = Relics(name, create = True)
  return relic
