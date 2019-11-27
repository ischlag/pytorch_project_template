import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence


def get_collate_fn(padding_value):
  def collate_pad_seq_classification(batch):
    (batch_x, batch_y) = zip(*batch)
    x = [torch.tensor(x) for x in batch_x]
    y = torch.tensor(batch_y).unsqueeze(-1)

    x_pad = pad_sequence(x, batch_first=True, padding_value=padding_value)
    y_pad = pad_sequence(y, batch_first=True, padding_value=padding_value)

    return x_pad, y_pad
  return collate_pad_seq_classification


class CustomIterator:
  def __init__(self, dataset, pad_value, data_loader_params):
    self.dataset = dataset
    self.pad_value = pad_value
    self.data_loader_params = data_loader_params
    self.iterator = None

  def init_dataloader(self):
    collate_fn = get_collate_fn(self.pad_value)
    self.iterator = iter(data.DataLoader(self.dataset,
                                         collate_fn=collate_fn,
                                         **self.data_loader_params))

  def __iter__(self):
    return self

  def __next__(self):
    if self.iterator is None:
      self.init_dataloader()

    try:
      return next(self.iterator)
    except StopIteration:
      self.iterator = None
      raise StopIteration