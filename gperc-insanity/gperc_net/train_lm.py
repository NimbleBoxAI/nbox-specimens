import os
import re
import numpy as np
from tqdm import trange
from newspaper import Article

import torch
from torch.optim import Adam
from torch.nn import functional as F

# -------
from gperc import TextConfig, Perceiver
from gperc.utils import set_seed

# -------

# ------------ create a dataset
def get_text():
    # To get text from the webpage(s)
    def get_text_from_url(url):
        article = Article(url)
        article.download()
        article.parse()
        return article.text

    fp = "/tmp/text.txt"
    if not os.path.exists(fp):
        print("Downloading text...")

        urls = [
            "https://towardsdatascience.com/lucy-says-hi-2031-agi-and-the-future-of-a-i-28b1e7b373f6",
            "https://towardsdatascience.com/to-do-great-data-science-embrace-domain-knowledge-167cb83dc050",
            "https://towardsdatascience.com/geometric-foundations-of-deep-learning-94cdd45b451d",
            "https://kitchingroup.cheme.cmu.edu/blog/2013/01/31/Smooth-transitions-between-discontinuous-functions/",
        ]
        text = "\n\n".join([get_text_from_url(u) for u in urls])

        # This Regex commands strips the texts so only alphabets, numbers, spaces and dot (.) remain
        text = re.sub(r"[^a-z0-9\s\.]", "", text.lower())

        with open(fp, "w") as f:
            f.write(text)

    else:
        with open(fp, "r") as f:
            text = f.read()

    return text


def get_tensors(text):
    # Get the set of unique words
    unq_words = sorted(list(set(text)))

    # Create vocabulary and inverse vocabulary to convert words in to numbers and numbers to words respectively
    vocabulary = {k: i for i, k in enumerate(unq_words)}
    vocabulary[len(vocabulary)] = "<mask>"

    # create dataset

    # Set sequence size, this is the number of words that are sent to the model at once
    seq_size = 256

    # Create buckets, basically blocks of length = sequence_size. We remove the last element of the bucket to ensure constant sizes
    buckets = [text[i : i + seq_size] for i in range(0, len(text), seq_size)][:-1]

    # [abcde fgh] x 208 samples
    # [123450678]
    input_ids = np.array([[vocabulary[token] for token in sequence] for sequence in buckets])
    t = torch.from_numpy(input_ids)
    return t, vocabulary


def create_dataset(tensor, mask_token_id, mask_perc=0.15):
    # create a boolen tensor with mask_perc being False
    mask = np.random.uniform(0, 1, tuple(tensor.shape)) < mask_perc
    _t = tensor.clone()
    _t[torch.tensor(mask)] = mask_token_id
    return _t


# ------------ create the model
class PerceiverMLM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.emb = torch.nn.Embedding(len(vocabulary), config.input_dim)
        # self.pos_emb = torch.nn.Parameter(torch.normal(mean=0, std=0.02, size=(config.input_len, config.input_dim)))
        self.perc = Perceiver(config)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # pos = torch.cat([self.pos_emb[None, ...] for _ in range(x.shape[0])], dim=0)
        # x = self.emb(x) + pos
        # print("!@#!@#$!@#$!@#$:", x.shape)
        # print(x[:1])
        logits = self.perc(x)
        return logits


# define the dataset
tensor_target, vocabulary = get_tensors(get_text())
mask_token_id = len(vocabulary) - 1
tensor_mask = create_dataset(tensor_target, mask_token_id, mask_perc=0.15)
assert tensor_mask.sum() != tensor_target.sum() and tensor_target.shape == tensor_mask.shape, "Your dataset function does not work!"

# define the config and the model
config = TextConfig(latent_dim=8, vocab_size=len(vocabulary), max_len=tensor_target.shape[1], latent_frac=0.15)
set_seed(config.seed)
print(config)
bert_model = PerceiverMLM(config)
print(bert_model.num_parameters())

# train
pbar = trange(4000)
optim = Adam(bert_model.parameters(), lr=0.001)
all_loss = []
all_acc = []

for i in pbar:
    optim.zero_grad()
    _y = bert_model(tensor_mask)
    _y = _y.contiguous().view(-1, _y.shape[-1])
    tensor_target = tensor_target.contiguous().view(-1)
    out = tensor_mask[tensor_mask == mask_token_id] == 0
    loss = F.cross_entropy(_y, tensor_target.long())

    all_loss.append(loss.item())
    all_acc.append((_y.argmax(dim=1) == tensor_target).sum().item() / len(tensor_target))
    pbar.set_description(f"loss: {np.mean(all_loss[-50:]):.4f} | acc: {all_acc[-1]:.4f}")
    loss.backward()

    torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
    optim.step()
