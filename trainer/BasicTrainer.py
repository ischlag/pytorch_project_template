#
#
#

import os
import time
import torch
import numpy as np
from utils.lib import assert_entries_exist, terminal_format, StopWatch
from utils.Map import Map


class BasicTrainer:
  def __init__(self, model, params, train_iterator, eval_iterator,
               optimizer, criterion, log):
    assert_entries_exist(params, ["log_every_n_steps",
                                  "eval_every_n_steps",
                                  "device",
                                  "log_folder"])
    self.p = params
    self.model = model.to(self.p.device)
    self.optimizer = optimizer
    self.criterion = criterion

    self.train_iterator = train_iterator
    self.eval_iterator = eval_iterator
    self.log = log

    self.state = Map()
    self.state.global_step = 0
    self.state.train_time = 0  # hours
    self.state.best_eval_loss = float("inf")
    self.state.best_eval_acc = 0
    self.state.best_train_time = 0  # hours
    self.state.best_step = 0

    self.best_eval_state_path = os.path.join(self.p.log_folder,
                                             "best_eval_state.pt")

  def _forward(self, x, y, voi):
    """ Compute a train/eval forward pass and update the variables
    of interest (VOI). """
    # compute logits and loss
    logits = self.model(x)
    loss = self.criterion(logits, y.view(-1))

    # compute VOIs
    accuracy = (torch.argmax(logits, dim=-1) == y.view(-1)).float().sum() \
                / len(y.view(-1))

    # track VOIs (make sure you don't track torch tensors)
    voi.losses.append(loss.item())
    voi.accuracies.append(accuracy.item())

    return loss

  def train(self):
    self.log("beginning training ...")
    self.model.train()
    self.optimizer.zero_grad()

    train_voi = Map()  # variables of interest
    train_voi.losses = []
    train_voi.accuracies = []
    train_voi.stopwatch = StopWatch()
    train_voi.batches = 0

    for x, y in self.train_iterator:
      # move batch to accelerator
      x = x.to(self.p.device)
      y = y.to(self.p.device)

      loss = self._forward(x, y, train_voi)

      # update weights
      loss.backward()
      self.optimizer.step()
      self.optimizer.zero_grad()
      self.state.global_step += 1
      train_voi.batches += 1

      # log train summaries
      if self.state.global_step % self.p.log_every_n_steps == 0:
        train_voi.stopwatch.stop()

        # compute summaries
        avg_loss = np.mean(train_voi.losses)
        avg_acc = np.mean(train_voi.accuracies)
        secs = train_voi.stopwatch.running_time
        batches_per_sec = train_voi.batches / secs
        hs = (secs / 60.0) / 60.0

        # keep track of training time
        self.state.train_time += hs

        # write terminal and file summaries
        vars = [
          ("train", ""),
          ("step", self.state.global_step, ":4"),
          ("loss", avg_loss, ":.5f"),
          ("acc", avg_acc, ":.3f"),
          ("b/s", batches_per_sec, ":.1f"),
          ("hours", hs, ":.1f")
        ]
        self.log(terminal_format(vars))

        # write tensorboard summaries
        # ...

        # clear
        train_voi.stopwatch.restart()
        train_voi.losses = []
        train_voi.accuracies = []

      # run evaluation
      if self.state.global_step % self.p.eval_every_n_steps == 0:
        train_voi.stopwatch.stop()
        self.evaluate(save_best=True, write_logs=True)
        self.model.train()
        train_voi.stopwatch.restart()

  def evaluate(self, save_best=False, iterator=None, write_logs=False):
    if iterator is None:
      iterator = self.eval_iterator

    self.model.eval()

    # variables of interest
    eval_voi = Map()
    eval_voi.losses = []
    eval_voi.accuracies = []
    eval_voi.batches = 0
    eval_voi.stopwatch = StopWatch()

    with torch.no_grad():
      for x, y in iterator:
        # move batch to accelerator
        x = x.to(self.p.device)
        y = y.to(self.p.device)

        # forward pass and track variables
        self._forward(x, y, eval_voi)
        eval_voi.batches += 1

    eval_voi.stopwatch.stop()

    # compute summaries
    avg_loss = np.mean(eval_voi.losses)
    avg_acc = np.mean(eval_voi.accuracies)
    secs = eval_voi.stopwatch.running_time
    batches_per_sec = eval_voi.batches / secs

    # track best summaries so far and save state/model
    if avg_loss < self.state.best_eval_loss:
      # new best model
      self.state.best_eval_loss = avg_loss
      self.state.best_eval_acc = avg_acc
      self.state.best_train_time = self.state.train_time
      self.state.best_step = self.state.global_step
      # save model
      if save_best and write_logs:
        self.save_state()

    # write terminal and file summaries
    vars = [
      ("eval", ""),
      ("loss", avg_loss, ":.5f"),
      ("acc", avg_acc, ":.3f"),
      ("b/s", batches_per_sec, ":2.1f"),
      ("| best", ""),
      ("loss", self.state.best_eval_loss, ":.5f"),
      ("acc", self.state.best_eval_acc, ":.3f"),
    ]
    self.log("\n")
    self.log(terminal_format(vars))

    # write tensorboard summaries
    # ...

  def save_state(self):
    curr_state = self.state
    curr_state.model = self.model.state_dict()
    curr_state.optimizer = self.optimizer.state_dict()
    #torch.save(obj=curr_state, f=self.best_eval_state_path)

  def load_state(self, path=None):
    if path is None:
      path = self.best_eval_state_path
    self.state = torch.load(path)
    self.model.load_state_dict(self.state.model)
    self.optimizer.load_state_dict(self.state.optimizer)
