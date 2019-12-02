import importlib

import torch
import torch.nn as nn
from sacred import Experiment
from munch import munchify

from utils.lib import import_and_populate
from utils.lib import setup_log_folder, save_current_script, setup_logger, \
  count_parameters

MODELS = "models"
TRAINERS = "trainers"
DATASETS = "datasets"

ex = Experiment("experiment")


@ex.config
def main_config():
  # we cannot use a Munch here due to sacred. Don't try.
  p = {}
  p["device"] = torch.device("cuda")
  p["n_gpus"] = torch.cuda.device_count()
  # data
  p["dataset_name"] = "bAbI_v1_2"
  p["dataset_variation"] = "bAbI10k"
  import_and_populate(DATASETS + "." + p["dataset_name"], p)
  p["train_batch_size"] = 32
  p["eval_batch_size"] = 64
  # optimizer
  p["learning_rate"] = 1e-3
  p["beta1"] = 0.9
  p["beta2"] = 0.999
  # model
  p["model_name"] = "seq_classification_lstm"
  import_and_populate(MODELS + "." + p["model_name"], p)
  # trainer
  p["trainer_name"] = "basic_trainer"
  import_and_populate(TRAINERS + "." + p["trainer_name"], p)
  # other
  folder_template = "logs/{}/{}/emb{}_h{}_l{}_lr{}_bs{}_gpus{}"
  folder_args = [
    p["dataset_variation"],
    p["model_name"],
    p["embedding_size"],
    p["hidden_size"],
    p["layers"],
    p["learning_rate"],
    p["train_batch_size"],
    p["n_gpus"],
  ]
  p["log_folder"] = folder_template.format(*folder_args)
  p["force"] = 0


@ex.automain
def run(p, _log):
  p = munchify(p)
  # setup log folder and backup source code
  if p.write_logs:
    setup_log_folder(p.log_folder)
    save_current_script(p.log_folder)

  # setup logger
  log, logger = setup_logger(p.log_folder if p.write_logs else None)
  log("{}".format(p))
  ex.logger = logger

  # import dataset
  log("load datasets ...")
  _module = importlib.import_module(DATASETS + "." + p.dataset_name)
  train_generator = _module.create_iterator(
    p=p,
    dataset_params={
      "variation": p.dataset_variation,
      "partition": "train",
    },
    data_loader_params={
      "batch_size": p.train_batch_size,
      "shuffle": p.train_shuffle,
      "num_workers": p.num_workers
    })

  eval_generator = _module.create_iterator(
    p=p,
    dataset_params={
      "variation": p.dataset_variation,
      "partition": "valid",
    },
    data_loader_params={
     "batch_size": p.eval_batch_size,
     "shuffle": p.eval_shuffle,
     "num_workers": p.num_workers
    })

  # build model
  log("load model ...")
  _module = importlib.import_module(MODELS + "." + p.model_name)
  model = _module.Model(p)
  log("{}".format(model))
  log("{} trainable parameters found. ".format(count_parameters(model)))

  # optimizer
  optimizer = torch.optim.Adam(params=model.parameters(),
                               lr=p.learning_rate,
                               betas=(p.beta1, p.beta2))

  # loss
  criterion = nn.CrossEntropyLoss(ignore_index=p.PAD)

  # DataParallel over multiple GPUs
  if p.n_gpus > 1:
    log("{} GPUs detected. Using nn.DataParallel. Batch-size per GPU: {}"
        .format(p.n_gpus, p.batch_size // p.n_gpus))
    model = nn.DataParallel(model)

  # create trainer
  log("load trainer ...")
  _module = importlib.import_module(TRAINERS + "." + p.trainer_name)
  trainer = _module.Trainer(model=model,
                            params=p,
                            train_generator=train_generator,
                            eval_generator=eval_generator,
                            optimizer=optimizer,
                            criterion=criterion,
                            log=log)

  # begin training
  log("start training ...")
  trainer.train()
