import os
import pickle
from torch.utils import data

bAbI10k_TEMPLATE = "en-valid-10k_{}.txt"
bAbI1k_TEMPLATE = "en-valid_{}.txt"
PARTITIONS = ["train", "valid", "test"]
DATA_PATH = "data/bAbI_v1.2"


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
