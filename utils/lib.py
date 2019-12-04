import importlib
import logging
import os
import shutil
import time
import pickle
import csv

def assert_entries_exist(map, keys):
  """ raises an attribute error if any on the keys does not exist. """
  for k in keys:
    if k not in map.__dict__.keys():
      raise AttributeError("Necessary parameter {} is missing!".format(k))


def count_parameters(model):
  """ returns the total number of parameters of a pytorch model. """
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def terminal_format(args):
  """
  args is a list of tuples and returns one string line.
  2-tuples are "{x[1]}".format(x[0]) i.e. value, format
  3-tuples are "{}={x[2]}".format(x[0],x[1]) i.e. label, value, format
  """
  line = ""
  for x in args:
    if len(x) == 3:
      line += ("{}={"+str(x[2])+"}").format(str(x[0]), x[1])
    elif len(x) == 2:
      line += ("{"+str(x[1])+"}").format(x[0])
    line += " "
  return line


def tf_add_scalars(writer, labels, scalars):
  """ Small helper function in order to perform multiple tensorboard
  write operations. """
  assert len(labels) == len(scalars)
  global_step = scalars[0]
  for i in range(1, len(labels)):
    writer.add_scalar(labels[i], scalars[i], global_step=global_step)
  writer.flush()


def setup_log_folder(path, force=0):
  """ Creates the folders necessary for path to exist. If a folder exists and
  force=0 (default) it asks for user input. force=1 always removes it.
  force=2 always keeps it. """
  if not os.path.exists(path):
    print("creating new log directory...")
    os.makedirs(path)
    return

  # folder already exists, request user input if not forced
  print("WARNING: The results directory ({}) already exists!".format(path))
  print("Delete previous results directory [y/n]? ", end="")
  if force == 0:
    choice = input()
    while choice not in ["y", "Y", "n", "N"]:
      print("invalid answer. try again.", end="")
      choice = input()
  elif force == 1:
    choice = "y"
    print(choice)
  elif force == 2:
    choice = "N"
    print(choice)

  if choice == "y" or choice == "Y":
    print("removing directory ...")
    shutil.rmtree(path)
    print("creating new log directory...")
    os.makedirs(path)


def save_current_script(log_folder):
  """ Takes all .py files in the current folder (getcwd) and saves them in
  log_folder/source_code. Assumes log_folder exists. Does NOT copy scripts
  if the source_code folder already exists. """
  source_folder = os.getcwd()
  target_folder = os.path.join(log_folder, "source_code")
  if os.path.exists(target_folder):
    # do not copy scripts if folder alreay exists
    return
  # create target folder
  os.makedirs(target_folder)
  # recursively copy all python files
  for path, _, file_names in os.walk("."):
    # skip log_folder itself or any folder with name logs
    if path.find(log_folder) >= 0 or path.find("logs") >= 0:
      continue
    for name in file_names:
      if name[-3:] == ".py":
        src_path = os.path.join(source_folder, path)
        trg_path = os.path.join(source_folder, target_folder, path)
        if not os.path.exists(trg_path):
          os.makedirs(trg_path)
        shutil.copyfile(src=os.path.join(src_path, name),
                        dst=os.path.join(trg_path, name))


def setup_logger(log_folder, file_name="output.log"):
  logger = logging.getLogger()
  logger.handlers = []
  logger.setLevel(logging.INFO)
  # create terminal handler
  s_handler = logging.StreamHandler()
  s_format = logging.Formatter('%(message)s')
  s_handler.setFormatter(s_format)
  logger.addHandler(s_handler)
  # create file handler
  if log_folder is not None:
    f_handler = logging.FileHandler(os.path.join(log_folder, file_name))
    f_format = logging.Formatter('%(asctime)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

  return lambda *x: logger.info((x[0].replace('{', '{{').replace('}', '}}')
                                 + "{} " * (len(x)-1)).format(*x[1:])), logger


def save_config(params, path, file_name):
    # dump sacred config to file if it doesn't exist
    conf_file = os.path.join(path, file_name)
    if not os.path.exists(conf_file):
      with open(conf_file, "wb+") as f:
        pickle.dump(params, f)


def import_and_populate(module_name, p):
  _module = importlib.import_module(module_name)
  return _module.load_default_params(p)


class CsvWriter:
  def __init__(self, column_names, path, file_name):
    self.csv_file = os.path.join(path, file_name)
    self.file = open(self.csv_file, "w+")
    self.writer = csv.writer(self.file)
    self.writer.writerow(column_names)

  def write(self, values):
    self.writer.writerow(values)
    self.file.flush()

  def close(self):
    self.file.close()


class StopWatch:
  """ Stops the time passed over several stops and restarts. Use flush to
  get the total difference since the last flush. """

  def __init__(self, running_time=0):
    self.running_time = running_time
    self.buffer_time = 0
    self.mark = time.time()

  def stop(self):
    if self.mark == 0:
      raise Exception("Stopwatch is not running. ")
    passed = time.time() - self.mark
    self.buffer_time += passed
    self.mark = 0

  def flush(self):
    buffer = self.buffer_time
    self.running_time += self.buffer_time
    self.buffer_time = 0
    return buffer

  def restart(self):
    if self.mark != 0:
      raise Exception("Stopwatch is already running?! ")
    self.mark = time.time()

  def reset(self):
    self.buffer_time = 0
    self.mark = 0

  def tick(self):
    self.stop()
    self.flush()
    self.restart()
