import os
from tempfile import gettempdir
from tqdm import trange
import numpy as np

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as TR
from torchvision.datasets import CIFAR10

from gperc import set_seed, build_position_encoding, ImageConfig, Perceiver

from nbox import operator, Lmao


class PerceiverCIFAR10(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.emb = build_position_encoding("trainable", config, 1024, 3)
    self.perceiver = Perceiver(config)

  def num_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

  def forward(self, x):
    pos_emb = torch.cat([self.emb[None, ...] for _ in range(x.shape[0])], dim=0)
    out = x + pos_emb
    return self.perceiver(out)


@operator()
def download_datasets():
  ds_train = CIFAR10(
    gettempdir(),
    # "./",
    train=True,
    download=True,
    transform=TR.Compose(
      [
        TR.ToTensor(),
        TR.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        TR.Lambda(lambda x: x.permute(1, 2, 0).reshape(-1, 3)),
      ]
    ),
  )
  ds_test = CIFAR10(
    gettempdir(),
    # "./",
    train=False,
    download=True,
    transform=TR.Compose(
      [
        TR.ToTensor(),
        TR.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        TR.Lambda(lambda x: x.permute(1, 2, 0).reshape(-1, 3)),
      ]
    ),
  )

  return ds_train, ds_test


@operator()
def create_model():
  # define the config and load the model
  config = ImageConfig(
    image_shape=[32, 32, 3],
    latent_len=32,
    latent_dim=32,
    n_classes=10,
  )
  set_seed(config.seed)
  model = PerceiverCIFAR10(config)
  print("model parameters:", model.num_parameters())
  return model


def save_model(model: torch.nn.Module, path: str, lmao: Lmao):
  print("saving model to", path)
  torch.save(model.state_dict(), path)
  lmao.save_file(path)

@operator()
def train_model(batch_size = 32, n_steps = 100, learning_rate = 0.001):
  ds_train, ds_test = download_datasets()
  model = create_model()
  lmao = Lmao(
    workspace_id="wnja9glc",
    project_name = "gperc_tests",
    metadata = {
      "model_config": model.config.get_dict(),
      "training_config": {
        "batch_size": batch_size,
        "n_steps": n_steps,
        "learning_rate": learning_rate,
      }
    }
  )

  # define the dataloaders, optimizers and lists
  dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
  dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False)
  iter_dl_train = iter(dl_train)

  pbar = trange(n_steps)
  optim = Adam(model.parameters(), lr = learning_rate)
  all_loss = []
  all_acc = []

  if not os.path.exists("all_checkpoints"):
    os.makedirs("all_checkpoints")

  # train!
  for i in pbar:
    try:
      x, y = next(iter_dl_train)
    except StopIteration:
      iter_dl_train = iter(dl_train)
      x, y = next(iter_dl_train)

    optim.zero_grad()
    _y = model(x)
    loss = F.cross_entropy(_y, y)
    all_loss.append(loss.item())
    all_acc.append((_y.argmax(dim=1) == y).sum().item() / len(y))
    # pbar.set_description(f"loss: {np.mean(all_loss[-50:]):.4f} | acc: {all_acc[-1]:.4f}")
    lmao.log({
      "train/loss": np.mean(all_loss[-50:]).item(),
      "train/acc": all_acc[-1],
    }, i)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()

    if (i + 1) % 50 == 0:
      model.eval()
      with torch.no_grad():
        _all_loss = []
        _all_acc = []
        for x, y in dl_test:
          _y = model(x)
          loss = F.cross_entropy(_y, y)
          _all_loss.append(loss.item())
          _all_acc.append((_y.argmax(-1) == y).sum().item() / len(y))
        print(f"Test Loss: {sum(_all_loss)} | Test Acc: {sum(_all_acc)/len(_all_acc)}")
        lmao.log({
          "test/loss": sum(_all_loss),
          "test/acc": sum(_all_acc)/len(_all_acc),
        }, i)
      if not os.path.exists(f"all_checkpoints/checkpoint-{i}"):
        os.makedirs(f"all_checkpoints/checkpoint-{i}")
      save_model(model, f"all_checkpoints/checkpoint-{i}/model.pt", lmao)
      lmao.save_file(f"all_checkpoints/checkpoint-{i}/")
      model.train()

  # in the end it doesn't even matter
  lmao.end()
