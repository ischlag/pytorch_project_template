import os
import torch
import numpy as np
from utils.lib import assert_entries_exist, terminal_format, StopWatch, \
                      add_scalars
from utils.Map import Map
from tensorboardX import SummaryWriter

BEST_MODEL_FILE_NAME = "best_eval_state.pt"
LAST_MODEL_FILE_NAME = "last_eval_state.pt"
TF_TRAIN_FOLDER_NAME = "train"
TF_EVAL_FOLDER_NAME = "valid"
NECESSARY_PARAMS = [
  "log_every_n_steps",
  "eval_every_n_steps",
  "device",
  "log_folder",
  "max_steps",  # -1 runs indefinitely
  "write_logs",
  "early_stopping_steps",  # -1 ignores early stopping
]


class BasicTrainer:
  """
  Trainer object that keeps track of the training state with some features:
  - saves best and last model after every evaluation
  - uses the log function to log terminal text
  - can restore the state of a training run and continue seamlessly
  - early stopping
  - saves last model whenever it logs (remove if model is big!)
  - saves best model whenever it evaluates
  """

  def __init__(self, model, params, train_generator, eval_generator,
               optimizer, criterion, log):
    assert_entries_exist(params, NECESSARY_PARAMS)
    self.p = params
    self.model = model.to(self.p.device)
    self.optimizer = optimizer
    self.criterion = criterion
    self.train_generator = train_generator
    self.train_iterator = iter(train_generator)
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
    self.state.script_running_time = 0

    # state paths
    self.best_eval_state_path = os.path.join(self.p.log_folder,
                                             BEST_MODEL_FILE_NAME)
    self.last_eval_state_path = os.path.join(self.p.log_folder,
                                             LAST_MODEL_FILE_NAME)
    # tensorflow events paths (tensorboard)
    self.tb_train_path = os.path.join(self.p.log_folder, TF_TRAIN_FOLDER_NAME)
    self.tb_eval_path = os.path.join(self.p.log_folder, TF_EVAL_FOLDER_NAME)
    self.train_writer = SummaryWriter(self.tb_train_path) if self.p.write_logs\
                                                          else None
    self.eval_writer = SummaryWriter(self.tb_eval_path) if self.p.write_logs\
                                                        else None

    # continue training where the last state ended (if it exists)
    if os.path.exists(self.last_eval_state_path):
      self.log("Previous model found! Reloading last state.")
      self.load_state(path=self.last_eval_state_path)

  def _forward(self, x, y, voi):
    """ Compute a train/eval forward pass and update the variables
    of interest (VOI). """
    # number of tokens that are not zero
    token_count = ((x == self.p.RA).nonzero()[:, 1] + 1).sum()
    # compute logits and loss
    logits = self.model(x)
    loss = self.criterion(logits, y.view(-1))

    # compute VOIs
    accuracy = (torch.argmax(logits, dim=-1) == y.view(-1)).float().sum() \
               / len(y.view(-1))

    # track VOIs (make sure you don't track torch tensors)
    voi.losses.append(loss.item())
    voi.accuracies.append(accuracy.item())
    voi.token_counts.append(token_count.item())

    return loss

  def train(self):
    self.model.train()
    self.optimizer.zero_grad()
    script_stopwatch = StopWatch(self.state.script_running_time)

    # variables of interest (don't forget to reset them after logging)
    train_voi = Map()
    train_voi.losses = []
    train_voi.accuracies = []
    train_voi.token_counts = []
    train_voi.stopwatch = StopWatch(self.state.train_time)
    train_voi.batches = 0

    while self.state.global_step < self.p.max_steps or self.p.max_steps == -1:
      # get next batch and reset iterator if epoch is over
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
      script_stopwatch.tick()

      # log train summaries
      if self.state.global_step % self.p.log_every_n_steps == 0:
        train_voi.stopwatch.stop()

        # compute summaries
        avg_loss = np.mean(train_voi.losses)
        avg_acc = np.mean(train_voi.accuracies)
        avg_token_count = np.mean(train_voi.token_counts)
        secs = train_voi.stopwatch.flush()
        batches_per_sec = train_voi.batches / secs
        tokens_per_sec = avg_token_count / secs
        hs = (script_stopwatch.running_time / 60.0) / 60.0

        # keep track of training time
        self.state.train_time = train_voi.stopwatch.running_time
        self.state.script_running_time = script_stopwatch.running_time

        # write terminal and file summaries
        vars = [
          ("train", ""),
          ("ep", self.state.epochs, ""),
          ("step", self.state.global_step, ":4"),
          ("loss", avg_loss, ":.5f"),
          ("acc", avg_acc, ":.3f"),
          ("t/s", tokens_per_sec, ":.1f"),
          ("hours", hs, ":.1f")
        ]
        self.log(terminal_format(vars))

        # write tensorboard summaries
        if self.p.write_logs:
          scalars = [
            ("loss", avg_loss),
            ("accuracy", avg_acc),
            ("batches_per_sec", batches_per_sec),
            ("tokens_per_sec", tokens_per_sec),
          ]
          add_scalars(self.train_writer, scalars, self.state.global_step)
          self.train_writer.flush()
          # restarts mess a little with tensorboard, saving the state here
          # would help to deal with that but it is a slow down for big models.
          # self.save_state(target=self.last_eval_state_path)

        # clear
        train_voi.stopwatch.restart()
        train_voi.losses = []
        train_voi.accuracies = []
        train_voi.token_counts = []
        train_voi.batches = 0

      # run evaluation
      if self.state.global_step % self.p.eval_every_n_steps == 0:
        train_voi.stopwatch.stop()
        self.evaluate(write_logs=self.p.write_logs)
        self.model.train()
        train_voi.stopwatch.restart()

        # check for early stopping
        steps_without_progress = self.state.global_step - self.state.best_step
        if self.p.early_stopping_steps >= 0 and \
           steps_without_progress > self.p.early_stopping_steps:
          self.log("No progress for {} steps".format(steps_without_progress))
          self.log("Stopping training.")
          return

  def evaluate(self, generator=None, write_logs=False):
    if generator is None:
      generator = self.eval_generator

    self.model.eval()

    # variables of interest
    eval_voi = Map()
    eval_voi.losses = []
    eval_voi.accuracies = []
    eval_voi.token_counts = []
    eval_voi.batches = 0
    eval_voi.stopwatch = StopWatch()

    with torch.no_grad():
      for x, y in generator:
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
    avg_token_count = np.mean(eval_voi.token_counts)
    secs = eval_voi.stopwatch.flush()
    batches_per_sec = eval_voi.batches / secs
    tokens_per_sec = avg_token_count / secs

    # track best summaries so far and save state/model
    if avg_loss < self.state.best_eval_loss and write_logs:
      # new best model
      self.state.best_eval_loss = avg_loss
      self.state.best_eval_acc = avg_acc
      self.state.best_train_time = self.state.train_time
      self.state.best_step = self.state.global_step
      # save best state so far
      self.save_state(target=self.best_eval_state_path)

    # save current state
    if write_logs:
      self.save_state(target=self.last_eval_state_path)

    # write terminal and file summaries
    vars = [
      ("eval", ""),
      ("loss", avg_loss, ":.5f"),
      ("acc", avg_acc, ":.3f"),
      ("t/s", tokens_per_sec, ":2.1f"),
      ("| best:", ""),
      ("loss", self.state.best_eval_loss, ":.5f"),
      ("acc", self.state.best_eval_acc, ":.3f"),
    ]
    self.log("")
    self.log(terminal_format(vars))
    # print folder path for easier identification of running experiments
    self.log("(" + self.p.log_folder + ")")
    self.log("")

    # write tensorboard summaries
    if write_logs:
      scalars = [
        ("loss", avg_loss),
        ("accuracy", avg_acc),
        ("batches_per_sec", batches_per_sec),
        ("tokens_per_sec", tokens_per_sec),
      ]
      add_scalars(self.eval_writer, scalars, self.state.global_step)
      self.eval_writer.flush()

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
    self.state = Map(curr_state["state"])
