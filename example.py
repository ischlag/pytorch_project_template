"""
%load_ext autoreload
%autoreload 2
"""
import torch
from torch.utils import data
from utils.Map import Map
import torch.nn as nn
import numpy as np
print(torch.cuda.is_available())
device = torch.device("cuda")
n_gpus = torch.cuda.device_count()

from utils.lib import setup_log_folder, save_current_script, setup_logger
from datasets.bAbI_v1_2 import bAbI10k, create_iterator
from models.sequence_classification_lstm import SeqClassifierLSTM
from trainer.BasicTrainer import BasicTrainer

# hyper-parameters
p = Map()
# data
p.num_workers = 1
p.shuffle = False
# optimizer
p.learning_rate=1e-3
p.batch_size = 32
# model
# ...
# other
# ...

p.device = device
p.n_gpus = n_gpus

# .... model
p.vocab_size = 180
p.embedding_size = 200
p.hidden_size = 256
p.layers = 2
# ... optimizer
p.learning_rate = 0.001
# ... trainer
p.log_every_n_steps = 25
p.eval_every_n_steps = 100
p.log_folder = "logs/"
p.max_steps = -1
p.write_logs = True
p.early_stopping_steps = -1

folder_template = "logs/{}/{}/emb{}_h{}_l{}_lr{}_bs{}_gpus{}"
folder_args = [
  "bAbI10k",
  "SeqClassifierLSTM",
  p.embedding_size,
  p.hidden_size,
  p.layers,
  p.learning_rate,
  p.batch_size,
  p.n_gpus,
]
p.log_folder = "logs/{}/{}/emb{}_h{}_l{}_lr{}_bs{}_gpus{}".format(*folder_args)


# setup log folder and backup source code
if p.write_logs:
  setup_log_folder(p.log_folder)
  save_current_script(p.log_folder)

# setup logger
log = setup_logger(p.log_folder if p.write_logs else None)

# import Dataset
train_dataset = bAbI10k(partition="train")
valid_dataset = bAbI10k(partition="valid")
idx2word = train_dataset.idx2word
p.PAD = train_dataset.word2idx["<pad>"]
p.RA = train_dataset.word2idx["<ra>"]

# build generators
data_loader_params = {k: p[k] for k in ["batch_size", "shuffle", "num_workers"]}
train_generator = create_iterator(train_dataset, p.PAD, data_loader_params)
valid_generator = create_iterator(valid_dataset, p.PAD, data_loader_params)

# build model
model = SeqClassifierLSTM(p)

# optimizer
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=p.learning_rate)

# loss
criterion = nn.CrossEntropyLoss(ignore_index=p.PAD)

# DataParallel over multiple GPUs
if p.n_gpus > 1:
  log("{} GPUs detected. Using nn.DataParallel. Batch-size per GPU: {}"
      .format(p.n_gpus, p.batch_size // n_gpus))
  model = nn.DataParallel(model)

trainer = BasicTrainer(model=model,
                       params=p,
                       train_generator=train_generator,
                       eval_generator=valid_generator,
                       optimizer=optimizer,
                       criterion=criterion,
                       log=log)

trainer.train()
