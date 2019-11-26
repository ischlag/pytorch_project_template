import time

def assert_entries_exist(map, keys):
  """ raises an attribute error if any on the keys does not exist. """
  for k in keys:
    if k not in map.__dict__.keys():
      raise AttributeError("Necessary parameter {} is missing!".format(k))


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


class StopWatch:
  def __init__(self):
    self.running_time = 0
    self.mark = time.time()

  def stop(self):
    if self.mark == 0:
      raise Exception("Stopwatch is not running. ")
    passed = time.time() - self.mark
    self.running_time += passed
    self.mark = 0

  def restart(self):
    if self.mark != 0:
      raise Exception("Stopwatch is already running?! ")
    self.mark = time.time()
