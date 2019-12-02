# Input: sequence
# Output: class
# Core: Custom LSTM implementation
#

import math
import torch
import torch.nn as nn
import torch.jit as jit

from typing import List, Tuple


def load_default_params(p):
    p["vocab_size"] = 180
    p["embedding_size"] = 200
    p["hidden_size"] = 256
    p["layers"] = 2


class Model(jit.ScriptModule):
  def __init__(self, params):
    super().__init__()
    self.p = p = params
    self.embedding = nn.Embedding(p.vocab_size, p.embedding_size)
    self.rnn = JITLSTM(input_size=p.embedding_size,
                       hidden_size=p.hidden_size,
                       n_layers=p.layers,
                       batch_first=True)
    self.linear = nn.Linear(p.hidden_size, p.vocab_size)

  def forward(self, x):
    # x_len = (x == self.p.RA).nonzero()[:, 1] + 1
    # x_len: [bsz]
    ra_pos = x == self.p.RA
    # ra_pos: [bsz, seqLen]
    x = self.embedding(x)
    # x: [bsz, seqLen, in_size]

    # loop
    out, out_state = self.rnn(x)
    # out: [bsz, seqLen, h_size]

    last_rnn_outputs = out[ra_pos, :]
    # last_rnn_outputs: [bsz, h_size]

    logits = self.linear(last_rnn_outputs)
    return logits


class JITLSTM(torch.jit.ScriptModule):
  __constants__ = ["timedim", "hidden_size", "n_layers", "cells"]

  def __init__(self, input_size, hidden_size, n_layers, batch_first=False):
    super().__init__()
    self.hidden_size = hidden_size
    self.n_layers = n_layers
    self.batch_first = batch_first
    self.timedim = 1 if self.batch_first else 0
    self.cells = torch.nn.ModuleList([
      LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
      for i in range(n_layers)])

  @torch.jit.script_method
  def do_forward(
    self,
    inputs: List[torch.Tensor],
    state: List[Tuple[torch.Tensor, torch.Tensor]]) \
    -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
    outputs = torch.jit.annotate(List[torch.Tensor], [])
    for t in range(len(inputs)):
      states = torch.jit.annotate(List[Tuple[torch.Tensor, torch.Tensor]], [])
      input = inputs[t]
      i = 0
      for cell in self.cells:
        input, new_state = cell(input, state[i])
        states += [new_state]
        i += 1
      outputs += [input]
      state = states
    return torch.stack(outputs, dim=self.timedim), state

  def forward(self, input: torch.Tensor, state: List[torch.Tensor] = None):
    inputs = input.unbind(self.timedim)
    batch_size = input.shape[0 if self.batch_first else 1]
    if state is None:
        state = [(torch.zeros(batch_size, self.hidden_size,
                              device=input.device, dtype=input.dtype) \
                  for _ in range(2)) for _ in range(self.n_layers)]
    return self.do_forward(inputs, state)


class LSTMCell(torch.jit.ScriptModule):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
    self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
    self.bias = nn.Parameter(torch.Tensor(4 * hidden_size))
    self.reset_parameters()

  def reset_parameters(self):
    std = 1.0 / math.sqrt(self.hidden_size)
    self.weight_hh.data.uniform_(-std, std)
    self.weight_ih.data.uniform_(-std, std)
    # Init forget gate bias to 1, rest to 0
    self.bias.data.fill_(0)
    self.bias.data[self.hidden_size:2*self.hidden_size] = 1

  @torch.jit.script_method
  def forward(self, input: torch.Tensor, state:Tuple[torch.Tensor, torch.Tensor]) -> \
          Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    hx, cx = state
    gates = torch.mm(input, self.weight_ih.t()) \
            + torch.mm(hx, self.weight_hh.t()) + self.bias
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, (hy, cy)
