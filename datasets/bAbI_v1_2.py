import importlib
import os
import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data

bAbI10k_TEMPLATE = "en-valid-10k_{}.txt"
bAbI1k_TEMPLATE = "en-valid_{}.txt"
PARTITIONS = ["train", "valid", "test"]
DATA_PATH = "data/bAbI_v1.2"


def load_default_params(p):
    p["dataset_variation"] = "bAbI10k"
    p["train_shuffle"] = True
    p["eval_shuffle"] = False
    p["num_workers"] = 6

def create_iterator(p, dataset_params, data_loader_params):
  """ Instantiates the dataset of this module according to the dataset
  parameters and creates a dataloader using data.Dataloader according to
  the data_loader_params arguments. """
  # instantiate one of the bAbI datasets
  _module = importlib.import_module("datasets.bAbI_v1_2")
  _class = getattr(_module, dataset_params["variation"])

  dataset = _class(partition=dataset_params["partition"])
  p.PAD = dataset.word2idx["<pad>"]
  p.RA = dataset.word2idx["<ra>"]

  # create data loader
  def collate_pad_seq_classification(batch):
    (batch_x, batch_y) = zip(*batch)
    x = [torch.tensor(x) for x in batch_x]
    y = torch.tensor(batch_y).unsqueeze(-1)

    x_pad = pad_sequence(x, batch_first=True, padding_value=p.PAD)
    y_pad = pad_sequence(y, batch_first=True, padding_value=p.PAD)

    return x_pad, y_pad

  return data.DataLoader(dataset,
                         collate_fn=collate_pad_seq_classification,
                         **data_loader_params)


def read_samples(file_path, word2idx):
  samples = []
  with open(file_path, "r") as f:
    for line in f:
      story, target = line.rstrip('\n').split("\t")
      words = story.split(" ")
      # encode samples
      x = [word2idx[word] for word in words]
      y = word2idx[target]
      samples.append((x, y))
  return samples


class bAbI10k(data.Dataset):
  def __init__(self, partition, folder=DATA_PATH):
    file_path = os.path.join(folder, bAbI10k_TEMPLATE.format(partition))

    with open(os.path.join(folder, "vocab.pkl"), "rb") as f:
      self.word2idx, self.idx2word = pickle.load(f)

    self.samples = read_samples(file_path, self.word2idx)

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    x, y = self.samples[index]
    return x, y


class bAbI1k(data.Dataset):
  def __init__(self, partition, folder=DATA_PATH):
    file_path = os.path.join(folder, bAbI1k_TEMPLATE.format(partition))

    with open(os.path.join(folder, "vocab.pkl"), "rb") as f:
      self.word2idx, self.idx2word = pickle.load(f)

    self.samples = read_samples(file_path, self.word2idx)

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    x, y = self.samples[index]
    return x, y


"""
d = bAbI1k(folder="../../data/bAbI_v1.2/", partition="train")
d.word2idx

it = iter(d)
batch = next(it)
batch[0]
batch[1]

" ".join(d.idx2word[idx] for idx in batch[0])
d.idx2word[batch[1]]
"""
