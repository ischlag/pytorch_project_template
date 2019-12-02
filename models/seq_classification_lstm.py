# Input: sequence
# Output: class
# Core: LSTM
#

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


def load_default_params(p):
    p["vocab_size"] = 180
    p["embedding_size"] = 200
    p["hidden_size"] = 256
    p["layers"] = 2


class Model(nn.Module):
  def __init__(self, params):
    super().__init__()
    self.p = p = params
    self.embedding = nn.Embedding(p.vocab_size, p.embedding_size)
    self.lstm = nn.LSTM(input_size=p.embedding_size,
                        hidden_size=p.hidden_size,
                        num_layers=p.layers)
    self.linear = nn.Linear(p.hidden_size, p.vocab_size)
    self.reset_parameters()

  def forward(self, x):
    x_len = (x == self.p.RA).nonzero()[:, 1] + 1
    # x_len: [bsz]
    ra_pos = x == self.p.RA
    # ra_pos: [bsz, seqLen]

    x = self.embedding(x)
    x = pack_padded_sequence(input=x,
                             lengths=x_len,
                             batch_first=True,
                             enforce_sorted=False)
    output_packed, _ = self.lstm(x)
    output_padded, _ = pad_packed_sequence(output_packed, batch_first=True)
    last_rnn_outputs = output_padded[ra_pos, :]
    logits = self.linear(last_rnn_outputs)
    return logits

  def reset_parameters(self):
    for layer in self.lstm.all_weights:
      # set bias_ih_l to 1
      layer[3].data[self.p.hidden_size:2 * self.p.hidden_size] = 1
