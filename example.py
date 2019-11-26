"""
%load_ext autoreload
%autoreload 2
"""
import torch
from torch.utils import data
import torch.nn as nn
import numpy as np
print(torch.cuda.is_available())
device = torch.device("cuda")
n_gpus = torch.cuda.device_count()

# hyperparams
from utils.Map import Map
p = Map()
# data
p.num_workers = 1
p.shuffle = False
# optimizer
p.learning_rate=1e-3
p.epochs=600
p.batch_size = 32
# model
p.h_size = 1024
# other
p.device = device
p.n_gpus = n_gpus
#p.log_folder = "logs/encoder_miniSprites/{}_h{}_lr{}_hw_last_dense_6".format(p.file_name, p.h_size, p.learning_rate)


# import Dataset
from datasets.bAbI_v1_2 import bAbI10k
train_dataset = bAbI10k(partition="train")
valid_dataset = bAbI10k(partition="valid")
idx2word = train_dataset.idx2word
p.PAD = train_dataset.word2idx["<pad>"]
p.RA = train_dataset.word2idx["<ra>"]

from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    (batch_x, batch_y) = zip(*batch)
    x = [torch.tensor(x) for x in batch_x]
    y = torch.tensor(batch_y).unsqueeze(-1)

    x_pad = pad_sequence(x, batch_first=True, padding_value=p.PAD)
    y_pad = pad_sequence(y, batch_first=True, padding_value=p.PAD)

    return x_pad, y_pad

data_loader_params = {k:p[k] for k in ["batch_size", "shuffle", "num_workers"]}
train_generator = data.DataLoader(train_dataset,
                                  collate_fn=collate_fn, **data_loader_params)
valid_generator = data.DataLoader(valid_dataset,
                                  collate_fn=collate_fn, **data_loader_params)

it = iter(train_generator)
x, y = next(it)

x.shape
y.shape


# build model
p.vocab_size = 180
p.embedding_size = 200
p.hidden_size = 256
p.layers = 2

from models.sequence_classification_lstm import SeqClassifierLSTM
model = SeqClassifierLSTM(p)

# optimizer
optimizer = torch.optim.Adam(params=model.parameters())


# loss
criterion = nn.CrossEntropyLoss(ignore_index=p.PAD)


# parallize
"""
# DataParallel over multiple GPUs
if n_gpus > 1:
  #log("{} GPUs detected. Using nn.DataParallel. Batch-size per GPU: {}"
  #      .format(n_gpus, p.batch_size // n_gpus))
  model = nn.DataParallel(model)
"""


from trainer.BasicTrainer import BasicTrainer
p.log_every_n_steps = 10
p.eval_every_n_steps = 100
p.log_folder = "logs/"
trainer = BasicTrainer(model=model,
                       params=p,
                       train_iterator=iter(train_generator),
                       eval_iterator=iter(valid_generator),
                       optimizer=optimizer,
                       criterion=criterion,
                       log=lambda x: print(x))


trainer.train()


def gen():
  for i in range(5):
    yield i


it = iter(gen())
next(it)
