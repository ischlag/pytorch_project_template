# Basic Trainer
#
# Features:
# - saves best/last model after evaluation
# - uses log function to log terminal text
# -

import os
import torch
import numpy as np
from utils.lib import assert_entries_exist, terminal_format, StopWatch
from utils.Map import Map

BEST_MODEL_FILE_NAME = "best_eval_state.pt"
LAST_MODEL_FILE_NAME = "last_eval_state.pt"
NECESSARY_PARAMS = [
  "log_every_n_steps",
  "eval_every_n_steps",
  "device",
  "log_folder",
  "max_steps"
]

class BasicTrainer:
  def __init__(self, model, params, train_generator, eval_generator,
               optimizer, criterion, log):
    assert_entries_exist(params, NECESSARY_PARAMS)
    self.p = params
    self.model = model.to(self.p.device)
    self.optimizer = optimizer
    self.criterion = criterion
    self.train_generator = train_generator
    self.train_iterator = iter(self.train_generator)
    self.eval_generator = eval_generator
    self.log = log

    # captures a restorable state
    self.state = Map()
    self.state.global_step = 0
    self.state.train_time = 0  # hours
    self.state.epochs = 0
    self.state.best_eval_loss = float("inf")
    self.state.best_eval_acc = 0
    self.state.best_train_time = 0  # hours
    self.state.best_step = 0

    self.best_eval_state_path = os.path.join(self.p.log_folder,
                                             BEST_MODEL_FILE_NAME)
    self.last_eval_state_path = os.path.join(self.p.log_folder,
                                             LAST_MODEL_FILE_NAME)

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

    while self.state.global_step < self.p.max_steps or self.p.max_steps == -1:
      # get next batch but reset iterator if epoch is over
      try:
        x, y = next(self.train_iterator)
      except StopIteration:
        self.state.epochs += 1
        self.train_iterator = iter(self.train_generator)

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
          ("ep", self.state.epochs, ""),
          ("step", self.state.global_step, ":4"),
          ("loss", avg_loss, ":.5f"),
          ("acc", avg_acc, ":.3f"),
          ("b/s", batches_per_sec, ":.1f"),
          ("hours", hs, ":.1f")
        ]
        self.log(terminal_format(vars))

        # write tensorboard summaries
        # TODO

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

  def evaluate(self, save_best=False, generator=None, write_logs=False):
    if generator is None:
      iterator = iter(self.eval_generator)
    else:
      iterator = iter(generator)

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
      # save best state so far
      if save_best and write_logs:
        self.save_state(target=self.best_eval_state_path)

    # save current state
    self.save_state(target=self.last_eval_state_path)

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
    self.log("")
    self.log(terminal_format(vars))
    self.log("")

    # write tensorboard summaries
    # TODO

  def save_state(self, target):
    # Apparently that Map() returns None for __getattr__ which results in an
    # error when one tries to pickle it.
    curr_state = {
      "state": self.state.__dict__,
      "model": self.model.state_dict(),
      "optimizer": self.optimizer.state_dict()
    }
    torch.save(obj=curr_state, f=target)

  def load_state(self, path=None):
    if path is None:
      path = self.best_eval_state_path
    curr_state = torch.load(path)
    self.model.load_state_dict(curr_state["model"])
    self.optimizer.load_state_dict(curr_state["optimizer"])
    self.state.__dict__ = curr_state["state"]
